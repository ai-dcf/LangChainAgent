"""从 notebooks/ 运行：python verify_chapter.py [--live]。不把输出写回 Notebook。"""
import argparse
import ast
import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys


def check_trace(trace, followup=False):
    from langchain.messages import AIMessage, ToolMessage
    from planning_lab import final_answer

    calls = [c for m in trace['messages'] if isinstance(m, AIMessage) for c in m.tool_calls]
    results = {m.tool_call_id: m for m in trace['messages'] if isinstance(m, ToolMessage)}
    assert calls, '没有真实工具调用'
    assert all(c['id'] in results for c in calls), '有工具调用缺少结果'
    assert all(results[c['id']].status != 'error' for c in calls), '出现工具错误，请检查原始轨迹'
    names = {c['name'] for c in calls}
    assert names <= {'write_todos', 'read_requirements', 'read_constraints'}, names
    assert 'write_todos' in names, '模型未使用待办工具'
    if not followup:
        assert {'read_requirements', 'read_constraints'} <= names, '模型未读取两份资料'
    snapshots = trace['snapshots']
    assert snapshots, '没有观察到 todos 状态更新'
    for todos in snapshots:
        assert all(isinstance(t['content'], str) and t['content'].strip() and
                   t['status'] in {'pending', 'in_progress', 'completed'} for t in todos)
    answer = final_answer(trace)
    assert isinstance(answer, str) and answer.strip(), '没有最终成果'
    # 这些只是基础词汇检查，不能代替语义复核。
    required = ['验收', '过期', '限流'] if followup else ['手机', '验证码', '密码', 'SQLite']
    assert all(word.lower() in answer.lower() for word in required), '最终回答缺少关键主题'
    content_sets = {tuple(t['content'] for t in todos) for todos in snapshots}
    constraint_index = next((i for i, m in enumerate(trace['messages'])
                             if isinstance(m, ToolMessage) and m.name == 'read_constraints'), None)
    before = after = None
    if constraint_index is not None:
        for i, message in enumerate(trace['messages']):
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls:
                if call['name'] == 'write_todos':
                    contents = tuple(t['content'] for t in call['args']['todos'])
                    if i < constraint_index:
                        before = contents
                    elif before is not None and contents != before:
                        after = contents
                        break
    observations = {
        'tool_calls': [c['name'] for c in calls],
        'todo_updates': len(snapshots),
        'todo_content_changed': len(content_sets) > 1,
        'content_changed_after_constraints': after is not None,
        'all_final_todos_completed': bool(trace['todos']) and
            all(t['status'] == 'completed' for t in trace['todos']),
        'snapshots': snapshots, 'answer': answer,
    }
    return observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='主动启用真实模型调用，会产生费用')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    os.chdir(here)
    sys.path.insert(0, str(here))
    from planning_lab import initialize
    initialize()
    data = here.parent / 'data'
    data.mkdir(exist_ok=True)
    report = {'mode': 'live' if args.live else 'static', 'complete': False,
              'versions': {n: importlib.metadata.version(n) for n in
                           ['langchain', 'langgraph', 'deepagents', 'langchain-deepseek']},
              'notebooks': [], 'warnings': []}
    target = data / f'verification-{report["mode"]}.json'

    def save():
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    try:
        for path in here.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))
        article = here.parent / 'LangChain-任务规划.md'
        for link in re.findall(r'\]\((\./[^)#]+)(?:#[^)]*)?\)', article.read_text(encoding='utf-8')):
            assert (article.parent / link).exists(), link
        notebooks = sorted(here.glob('*.ipynb'))
        assert len(notebooks) == 3
        for path in notebooks:
            nb = json.loads(path.read_text(encoding='utf-8'))
            env = {'__name__': '__main__'}  # 独立命名空间，不共享前一本的变量。
            output = io.StringIO()
            count = 0
            with contextlib.redirect_stdout(output):
                for index, cell in enumerate(nb['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    assert cell['execution_count'] is None and cell['outputs'] == []
                    source = ''.join(cell['source'])
                    ast.parse(source)
                    # 只覆盖明确的教学开关，不修改交付文件。
                    if args.live:
                        source = source.replace('RUN_LIVE = False', 'RUN_LIVE = True')
                    exec(compile(source, f'{path.name}:cell-{index}', 'exec'), env)
                    count += 1
            entry = {'name': path.name, 'code_cells': count, 'passed': True}
            if args.live:
                entry['first_turn'] = check_trace(env['trace'])
                if not entry['first_turn']['content_changed_after_constraints']:
                    report['warnings'].append(path.name + '：未观察到读取约束后待办内容改变')
                if not entry['first_turn']['all_final_todos_completed']:
                    report['warnings'].append(path.name + '：最终待办未全部完成')
                if 'follow_trace' in env:
                    entry['followup'] = check_trace(env['follow_trace'], followup=True)
                    if not entry['followup']['all_final_todos_completed']:
                        report['warnings'].append(path.name + '：第二轮待办未全部完成')
                    entry['same_saver_restored_and_new_thread_empty'] = True
                # 真实过程只存本机忽略目录，Notebook 不保存执行输出。
                (data / (path.stem + '-live.txt')).write_text(output.getvalue(), encoding='utf-8')
            report['notebooks'].append(entry)
            save()
            print(f'PASS {path.name}: {count} code cells', flush=True)
        report['complete'] = True
        # 静态模式没有观察真实规划行为，不能据此声称预期行为满足。
        report['expected_observations_met'] = not report['warnings'] if args.live else None
        save()
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__,
                             'notebook': path.name if 'path' in locals() else None,
                             'cell': index if 'index' in locals() else None}
        save()
        # 不输出供应商异常正文，避免请求配置等敏感内容进入日志。
        print('FAILED:', report['failure'], file=sys.stderr)
        raise SystemExit(1) from None
    for warning in report['warnings']:
        print('观察未满足：', warning)
    print('报告：', target.relative_to(here.parent))
    if report['warnings']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
