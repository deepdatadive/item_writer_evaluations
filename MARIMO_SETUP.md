# Marimo Setup Guide

## What Was Done

Your Jupyter notebooks have been successfully converted to [Marimo](https://marimo.io), a modern reactive Python notebook system!

### Converted Notebooks

1. ✅ `experiments/basic_vs_ai_assist/notebooks/basic_vs_ai_assist.py`
2. ✅ `experiments/processing_block_size/notebooks/processing_block_size.py`

### Updated Files

- ✅ `requirements.in` - Added marimo
- ✅ `requirements.txt` - Compiled with marimo and all dependencies
- ✅ `README.md` - Added Marimo documentation
- ✅ `experiments/basic_vs_ai_assist/QUICKSTART.md` - Updated with Marimo instructions
- ✅ `experiments/processing_block_size/QUICKSTART.md` - Updated with Marimo instructions

## How to Use Marimo

### Running a Notebook (Interactive Mode)

```bash
cd /home/chris-foster/PycharmProjects/item_writer_evaluations
source venv/bin/activate
cd experiments/basic_vs_ai_assist/notebooks
marimo edit basic_vs_ai_assist.py
```

This opens the notebook in your browser where you can:
- Edit and run cells interactively
- Automatic reactivity - cells update when dependencies change
- No hidden state issues
- Better debugging experience

### Running as a Web App

```bash
marimo run basic_vs_ai_assist.py
```

This runs the notebook as a read-only web application - perfect for sharing results!

### Running the Tutorial

```bash
marimo tutorial intro
```

This launches Marimo's interactive tutorial to learn the basics.

## Why Marimo?

### Key Advantages

1. **Reactive Execution** 📊
   - Cells automatically re-run when their dependencies change
   - No more "run all" needed
   - Always in sync

2. **No Hidden State** 🔒
   - Reproducibility built-in
   - No out-of-order execution issues
   - Variables are always fresh

3. **Pure Python Files** 📝
   - Better for version control (git diff works properly)
   - Can be run as scripts
   - Easy code review

4. **Better UI** 🎨
   - Modern, clean interface
   - Built-in interactive widgets
   - Dark mode support
   - Better error messages

5. **Dual Mode** 🚀
   - Edit mode: Interactive development
   - Run mode: Deployable web app
   - No separate deployment needed

### Comparison with Jupyter

| Feature | Jupyter | Marimo |
|---------|---------|--------|
| Reactive execution | ❌ Manual | ✅ Automatic |
| Hidden state | ⚠️ Possible | ✅ Prevented |
| File format | JSON | Pure Python |
| Git-friendly | ⚠️ Difficult | ✅ Easy |
| Interactive widgets | ✅ Good | ✅ Better |
| Deployment | Complex | Built-in |
| Learning curve | Low | Low-Medium |

## Converting More Notebooks

To convert any Jupyter notebook to Marimo:

```bash
marimo convert your_notebook.ipynb -o your_notebook.py
```

## Using Both Jupyter and Marimo

Both formats are maintained in this project:
- `.ipynb` files for Jupyter
- `.py` files for Marimo

You can use whichever you prefer! The Marimo files are automatically kept in sync when converted.

## Tips for Marimo

1. **Cell Dependencies**: Marimo automatically tracks which cells depend on which variables
2. **UI Elements**: Use `mo.ui` for interactive widgets (sliders, dropdowns, etc.)
3. **Markdown**: Use `mo.md()` for formatted text
4. **Debugging**: Click on variables to inspect them in the UI
5. **Keyboard Shortcuts**: Press `?` in the editor for shortcuts

## Common Commands

```bash
# Edit a notebook
marimo edit notebook.py

# Run as web app
marimo run notebook.py

# Run as web app on specific port
marimo run notebook.py --port 8080

# Convert Jupyter to Marimo
marimo convert notebook.ipynb

# Create new notebook
marimo new notebook.py

# Tutorial
marimo tutorial intro

# Help
marimo --help
```

## Next Steps

1. Try running the basic_vs_ai_assist notebook:
   ```bash
   cd experiments/basic_vs_ai_assist/notebooks
   marimo edit basic_vs_ai_assist.py
   ```

2. Explore the interactive features:
   - Change a cell and watch dependent cells update
   - Try the built-in widgets
   - Export as HTML

3. Run the tutorial:
   ```bash
   marimo tutorial intro
   ```

## Resources

- **Official Website**: https://marimo.io
- **Documentation**: https://docs.marimo.io
- **GitHub**: https://github.com/marimo-team/marimo
- **Examples**: https://marimo.io/gallery

## Questions?

Marimo has excellent documentation and a helpful community. Check out:
- The official docs at docs.marimo.io
- The GitHub discussions
- The examples gallery

Enjoy your reactive notebooks! 🎉

