# Multimodal RAG Paper

LaTeX project for the Multimodal RAG Disaster Impact Assessment paper.

## Structure

```
Multimodal-RAG-Paper/
├── main.tex                 # Main document
├── neurips_2024.sty         # NeurIPS style (download official from NeurIPS)
├── references.bib           # Bibliography
├── sections/
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── analysis.tex
│   ├── conclusion.tex
│   └── appendix.tex
├── tables/
│   ├── dataset_stats.tex
│   ├── judges.tex
│   ├── models.tex
│   ├── main_results.tex
│   ├── category_results.tex
│   ├── judge_scores.tex
│   ├── judge_agreement.tex
│   ├── human_ai_agreement.tex
│   ├── ablation.tex
│   ├── retrieval_metrics.tex
│   └── full_results.tex
└── figures/
    └── (add figures here)
```

## Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use `latexmk`:
```bash
latexmk -pdf main.tex
```

## TODO Items

Search for `\todo{...}` in the source files to find placeholders that need to be filled in.

### Priority TODOs:
1. Fill in actual numbers in result tables
2. Add system architecture diagram
3. Complete introduction and related work
4. Download official NeurIPS 2024 style file

## Key Tables to Fill

| Table | Location | Data Source |
|-------|----------|-------------|
| Main Results | `tables/main_results.tex` | Run evaluation scripts |
| Judge Scores | `tables/judge_scores.tex` | From `multi_judge_validation_25.json` |
| Human-AI Agreement | `tables/human_ai_agreement.tex` | After co-worker annotation |

## Notes

- Paper uses NeurIPS 2024 style (preprint mode)
- System name: `\systemname{}` → HarveyRAG
- All `\todo{}` markers are highlighted in red
