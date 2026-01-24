"""
代码覆盖率检查工具使用介绍：

1. 检查所有算子测例：
   python scripts/check_coverage.py

2. 检查单个或多个指定算子：
   python scripts/check_coverage.py kop_acos
   python scripts/check_coverage.py kop_acos kop_gru

3. 可选参数：
   --no-html    不生成 HTML 报告（默认自动生成到 coverage_report/ 目录）
   --no-branch  禁用分支覆盖率统计（默认启用）

注意：生成的 HTML 报告中，网页文件位于 html/ 目录，静态资源位于 static/ 目录。
"""

import argparse
import sys
import shutil
from pathlib import Path
import runpy

import coverage


PROJECT_ROOT = Path(__file__).parent.parent
SRC_PARENT = PROJECT_ROOT / "src"
SRC_DIR = SRC_PARENT / "kp_onnx_ssbo"                   # 需要检查覆盖率的目录
TEST_DIR = PROJECT_ROOT / "src" / "kp_onnx_test"        # 存放测试文件的目录


def get_test_file(module_name: str) -> str:
    """根据模块名获取测试文件名"""
    if module_name.startswith("kop_"):
        return f"{module_name}_test.py"
    return f"kop_{module_name}_test.py"


def run_tests_with_coverage(test_files, modules=None, branch=True, html=True) -> bool:
    """在当前进程中运行测例并收集覆盖率"""
    # 确保 src 在 sys.path 中，便于导入 kp_onnx
    if str(SRC_PARENT) not in sys.path:
        sys.path.insert(0, str(SRC_PARENT))

    report_root = PROJECT_ROOT / "coverage_report"
    static_dir = report_root / "static"
    html_dir = report_root / "html"
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # 将 .coverage 原始数据文件存放在 static 目录下
    data_file = static_dir / ".coverage"

    print("=" * 80)
    print("开始运行测试并收集代码覆盖率...")
    print("=" * 80)

    cov = coverage.Coverage(source=[str(SRC_DIR)], branch=branch, data_file=str(data_file))
    cov.start()

    all_passed = True
    for test_file in test_files:
        test_path = TEST_DIR / test_file
        if not test_path.exists():
            print(f"警告: 测试文件不存在: {test_path}")
            all_passed = False
            continue

        print(f"\n运行测试: {test_file}")
        print("-" * 80)
        try:
            runpy.run_path(str(test_path), run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None):
                print(f"测试脚本退出码: {e.code}")
                all_passed = False
        except Exception as e:
            print(f"执行 {test_file} 时发生异常: {e}")
            all_passed = False

    cov.stop()
    cov.save()

    print("\n" + "=" * 80)
    print("代码覆盖率报告")
    print("=" * 80)

    include = None
    if modules:
        include = []
        for m in modules:
            name = m if m.startswith("kop_") else f"kop_{m}"
            include.append(str(SRC_DIR / f"{name}.py"))
        print(f"\n只统计以下模块: {', '.join(modules)}")

    # 终端报告
    cov.report(include=include, show_missing=True)

    # HTML 报告
    if html:
        temp_dir = report_root / "temp"
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        
        cov.html_report(directory=str(temp_dir), include=include)
        
        # 准备/清理目录
        if html_dir.exists(): shutil.rmtree(html_dir)
        html_dir.mkdir(parents=True)
        # static 目录已在开头创建，只需确保它是干净的（除了我们刚生成的 .coverage）
        # 这里为了简单起见，不删除 static 目录，只在移动文件时处理
            
        assets = []
        for f in temp_dir.iterdir():
            if f.name in ("class_index.html", "function_index.html"):
                continue
            if f.suffix == ".html":
                shutil.move(str(f), str(html_dir / f.name))
            elif f.name != ".gitignore":
                assets.append(f.name)
                # 如果 static 目录下已存在同名资产文件，先删除
                dest = static_dir / f.name
                if dest.exists(): dest.unlink()
                shutil.move(str(f), str(dest))
                
        for html_file in html_dir.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            for asset in assets:
                content = content.replace(f'href="{asset}', f'href="../static/{asset}')
                content = content.replace(f'src="{asset}', f'src="../static/{asset}')
                content = content.replace(f'"{asset}"', f'"../static/{asset}"')
            html_file.write_text(content, encoding="utf-8")
            
        shutil.rmtree(temp_dir)
        print(f"\nHTML 报告生成于: {html_dir / 'index.html'}")
        print(f"原始数据文件位于: {data_file}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="代码覆盖率检查工具")
    parser.add_argument(
        "modules",
        nargs="*",
        help="要检查的模块名（如 kop_gru, kop_col2im），不指定则检查所有"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="不生成 HTML 报告（默认会生成）"
    )
    parser.add_argument(
        "--no-branch",
        action="store_true",
        help="关闭分支覆盖率（默认开启）"
    )

    args = parser.parse_args()

    # 确定需要跑的测试文件
    if args.modules:
        test_files = [get_test_file(m) for m in args.modules]
    else:
        test_files = sorted(f.name for f in TEST_DIR.glob("kop_*_test.py"))
        print(f"将检查所有测试文件: {len(test_files)} 个")

    success = run_tests_with_coverage(
        test_files=test_files,
        modules=args.modules or None,
        branch=not args.no_branch,
        html=not args.no_html,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
