"""从 notebooks/ 运行：python verify_chapter.py [--live]。

在临时 data 子目录中逐个执行代码单元，不把执行结果写回 Notebook。
--live 需要现有 .env，额外启用小规模真实调用；不启用整文件真实分块分析。
"""
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
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    os.chdir(here)
    sys.path.insert(0, str(here))
    import context_lab
    context_lab.initialize()
    report = {'mode': 'live' if args.live else 'offline',
              'versions': {name: importlib.metadata.version(name) for name in
                           ['langchain', 'langgraph', 'langgraph-checkpoint-sqlite']},
              'complete': False, 'notebooks': []}
    output = here.parent / 'data' / f'verification-{report["mode"]}.json'
    def save_report():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save_report()  # 新运行立即标记未完成，失败时不能留下旧的“全部通过”报告。
    os.environ['CH09_RUN_LIVE'] = '1' if args.live else '0'
    os.environ['CH09_LIVE_MAP'] = '0'
    with tempfile.TemporaryDirectory(prefix='verify-', dir='../data') as tmp:
        # Notebook 仍从规定目录运行，只把运行数据重定向到临时目录。
        original = context_lab.initialize
        context_lab.initialize = lambda: Path(tmp)
        try:
            for path in sorted(here.glob('*.ipynb')):
                nb = json.loads(path.read_text(encoding='utf-8'))
                env = {'__name__': '__main__'}
                stream = io.StringIO()
                cell_count = 0
                with contextlib.redirect_stdout(stream):
                    for index, cell in enumerate(nb['cells']):
                        if cell['cell_type'] != 'code':
                            continue
                        assert cell['execution_count'] is None and cell['outputs'] == []
                        source = ''.join(cell['source'])
                        ast.parse(source)
                        try:
                            exec(compile(source, f'{path.name}:cell-{index}', 'exec'), env)
                        except Exception as exc:
                            report['failure'] = {'notebook': path.name, 'cell': index,
                                                 'type': type(exc).__name__, 'error': str(exc)}
                            save_report()
                            print(stream.getvalue()[-3000:], file=sys.stderr)
                            print(f'FAILED {path.name} cell {index}', file=sys.stderr)
                            raise
                        cell_count += 1
                entry = {'name': path.name, 'code_cells': cell_count, 'passed': True,
                         'output': stream.getvalue()}
                report['notebooks'].append(entry)
                save_report()
                print(f'PASS {path.name}: {cell_count} code cells', flush=True)
        finally:
            context_lab.initialize = original
    article = here.parent / 'LangChain-上下文管理.md'
    if article.stat().st_size:
        text = article.read_text(encoding='utf-8')
        for target in re.findall(r'\]\((\./[^)#]+)(?:#[^)]*)?\)', text):
            assert (article.parent / target).exists(), target
    report['complete'] = True
    save_report()
    print('Report:', output.relative_to(here.parent))


if __name__ == '__main__':
    main()
