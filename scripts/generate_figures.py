import graphviz
import os

# Ensure output directory exists
output_dir = "paper/figures"
os.makedirs(output_dir, exist_ok=True)

def generate_figure_1():
    """Generate System Architecture Diagram"""
    dot = graphviz.Digraph(comment='System Architecture', format='png')
    dot.attr(rankdir='LR', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', fontname='Helvetica')
    
    # Subgraph for Data Sources
    with dot.subgraph(name='cluster_0') as c:
        c.attr(style='dashed', label='Multimodal Data Sources', color='gray')
        c.node('Sat', 'Satellite Imagery\n(Geo-Indexed Tiles)', fillcolor='#E3F2FD')
        c.node('Text', 'Social Media & Calls\n(Tweets, 311)', fillcolor='#FFF3E0')
        c.node('Sensor', 'Sensor Data\n(Rain Gauges)', fillcolor='#E8F5E9')
        c.node('FEMA', 'FEMA Knowledge Base\n(Flood Depth Grid)', fillcolor='#F3E5F5')

    # Retrieval System
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Retriever Module', color='blue')
        c.node('Hybrid', 'Hybrid Retrieval\n(Dense + Sparse)')
        c.node('Geo', 'Geo-Spatial Filter\n(Radius Search)')
        
    # App Logic
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Core Application', color='black')
        c.node('Split', 'Split Pipeline\nOrchestrator')
        c.node('Fusion', 'Fusion Engine\n(Visual Additive)')

    # Output
    dot.node('Output', 'Final Assessment\n(JSON + Evidence)', shape='ellipse', style='filled', fillcolor='gold')

    # Edges
    dot.edge('Sat', 'Geo')
    dot.edge('Text', 'Hybrid')
    dot.edge('Sensor', 'Hybrid')
    dot.edge('FEMA', 'Hybrid')
    
    dot.edge('Geo', 'Split')
    dot.edge('Hybrid', 'Split')
    
    dot.edge('Split', 'Fusion')
    dot.edge('Fusion', 'Output')

    output_path = os.path.join(output_dir, 'system_architecture')
    dot.render(output_path, cleanup=True)
    print(f"Generated {output_path}.png")

def generate_figure_2():
    """Generate Split Pipeline Diagram"""
    dot = graphviz.Digraph(comment='Split Pipeline', format='png')
    dot.attr(rankdir='TB', dpi='300') # Top to Bottom
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', fontname='Helvetica')

    # Input
    dot.node('Input', 'Retrieved Context\n(Text + Imagery + Sensors)', shape='parallelogram', fillcolor='lightgray')

    # Branches
    with dot.subgraph(name='cluster_text') as c:
        c.attr(label='Text Analyst (LLM)', color='orange', style='dashed')
        c.node('AnalysisT', 'Analyze Text Reports\n(Tweets/Calls/Gauges)')
        c.node('EstT', 'Text Estimate\n(Hazard & Damage)')

    with dot.subgraph(name='cluster_visual') as c:
        c.attr(label='Visual Analyst (VLM)', color='blue', style='dashed')
        c.node('AnalysisV', 'Analyze Imagery\n(Flood Extent/Debris)')
        c.node('EstV', 'Visual Estimate\n(Confirmation Signal)')

    # Fusion
    dot.node('Fusion', 'Visual Additive Fusion\nScore = Text + λ * Visual', shape='hexagon', fillcolor='#D1C4E9')
    
    # Decisions
    dot.node('Check', 'Visual Confirmation?', shape='diamond', style='filled', fillcolor='white')
    
    # Output
    dot.node('Final', 'Unified Risk Score', shape='ellipse', fillcolor='gold')

    # Edges
    dot.edge('Input', 'AnalysisT')
    dot.edge('Input', 'AnalysisV')
    
    dot.edge('AnalysisT', 'EstT')
    dot.edge('AnalysisV', 'EstV')
    
    dot.edge('EstT', 'Fusion')
    dot.edge('EstV', 'Check')
    dot.edge('Check', 'Fusion', label='Additive Boost')
    
    dot.edge('Fusion', 'Final')

    output_path = os.path.join(output_dir, 'split_pipeline')
    dot.render(output_path, cleanup=True)
    print(f"Generated {output_path}.png")

if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
