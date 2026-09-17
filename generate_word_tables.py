"""生成 Word 格式表格

将 all_tables_data.json 中的表格数据转换为 Word 文档。
"""

import os
import json

try:
    from docx import Document
    from docx.shared import Pt, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import nsdecls
    from docx.oxml import parse_xml
except ImportError:
    print("请先安装 python-docx: pip install python-docx")
    exit(1)


def add_table_to_doc(doc, table_data, table_title):
    """将表格数据添加到 Word 文档"""
    if not table_data:
        return
    
    # 添加标题
    doc.add_heading(table_title, level=2)
    
    # 创建表格
    headers = list(table_data[0].keys())
    num_cols = len(headers)
    num_rows = len(table_data) + 1  # +1 for header
    
    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # 设置表头
    header_row = table.rows[0]
    for i, header in enumerate(headers):
        cell = header_row.cells[i]
        cell.text = header
        # 加粗表头
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # 灰色背景
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="D9D9D9"/>')
        cell._tc.get_or_add_tcPr().append(shading)
    
    # 填充数据
    for row_idx, row_data in enumerate(table_data):
        row = table.rows[row_idx + 1]
        for col_idx, header in enumerate(headers):
            cell = row.cells[col_idx]
            cell.text = str(row_data[header])
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 添加空行
    doc.add_paragraph()


def main():
    """生成 Word 表格文档"""
    
    # 加载 JSON 数据
    json_path = 'outputs/paper/all_tables_data.json'
    if not os.path.exists(json_path):
        print(f"错误：找不到 {json_path}")
        print("请先运行 generate_tables.py")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        all_tables = json.load(f)
    
    # 创建 Word 文档
    doc = Document()
    
    # 设置文档标题
    doc.add_heading('论文表格数据', level=0)
    doc.add_paragraph('以下为实验生成的真实数据，可直接复制到论文中。')
    doc.add_paragraph()
    
    # 添加各表格
    table_configs = [
        ('table1_alns_config', '表1 ALNS参数自适应配置'),
        ('table1_greedy_selection', '表1 Greedy-EDF任务选择过程示例（N=6，seed=11）'),
        ('table2_greedy_events', '表2 Greedy-EDF算法执行关键事件（N=6，seed=11）'),
        ('table3_lower_bound', '表3 下界计算示例（N=6）'),
        ('table4_pruning_stats', '表4 剪枝策略效果统计（N=6）'),
        ('table5_initial_solution', '表5 初始解分配情况（N=10）'),
        ('table6_operator_weights', '表6 算子权重演化过程（N=10）'),
        ('table7_acceptance_stats', '表7 解接受情况统计（N=10）'),
    ]
    
    for key, title in table_configs:
        if key in all_tables:
            add_table_to_doc(doc, all_tables[key], title)
            print(f"[✓] 已添加：{title}")
        else:
            print(f"[!] 跳过（无数据）：{title}")
    
    # 保存文档
    import time as tm
    timestamp = tm.strftime('%H%M%S')
    output_path = f'outputs/paper/论文表格_{timestamp}.docx'
    doc.save(output_path)
    print(f"\n[✓] Word 文档已保存: {output_path}")


if __name__ == '__main__':
    main()
