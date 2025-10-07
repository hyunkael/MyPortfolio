# Deep Learning Portfolio

A comprehensive portfolio showcasing my journey through Deep Learning coursework, including laboratory activities, exercises, and projects.

## 📚 About This Portfolio

This Jupyter Book portfolio documents my learning progress in Deep Learning and Artificial Intelligence. It contains:

- **Laboratory Activities**: Hands-on experiments exploring fundamental and advanced deep learning concepts
- **Exercises**: Problem-solving exercises that reinforce theoretical knowledge
- **Projects**: End-to-end projects demonstrating real-world applications
- **Reflections**: Personal insights and learnings throughout the journey

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://hyunkael.github.io/DLPortfolio
   cd DLPortfolio
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Building the Book

To build the Jupyter Book locally:

```bash
jupyter-book build .
```

The built HTML files will be in the `_build/html` directory. Open `_build/html/index.html` in your browser to view the book.

### Cleaning Build Files

To clean previous builds:

```bash
jupyter-book clean .
```

To clean and remove all build files:

```bash
jupyter-book clean . --all
```

## 📂 Project Structure

```
DLPortfolio/
├── _config.yml              # Jupyter Book configuration
├── _toc.yml                 # Table of contents
├── intro.md                 # Main landing page
├── about.md                 # About me page
├── reflections.md           # Course reflections
├── requirements.txt         # Python dependencies
├── references.bib           # Bibliography
├── README.md               # This file
├── _static/                # Static assets
│   └── custom.css          # Custom CSS styling
├── labs/                   # Laboratory activities
│   ├── index.md
│   └── lab1.ipynb
├── exercises/              # Practice exercises
│   ├── index.md
│   └── placeholder.md
└── projects/               # Major projects
    ├── index.md
    └── placeholder.md
```

## 🛠️ Technologies Used

- **Jupyter Book**: Static site generator for computational narratives
- **Python**: Primary programming language
- **NumPy & Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **GitHub Pages**: Free hosting for the portfolio
- **MyST Markdown**: Enhanced markdown for technical writing

## 🤝 Contributing

This is a personal learning portfolio, but suggestions and feedback are welcome! Feel free to:
- Open an issue for suggestions
- Submit a pull request for corrections
- Share your own learning journey

## 📧 Contact

- **Name**: Qylle Christian Quiño
- **Email**: quinoqylle6@gmail.com
- **GitHub**: [@hyunkael](https://github.com/hyunkael)

---

## 🔄 Continuous Updates

This portfolio is continuously updated as I progress through the Deep Learning course. Check back regularly for new content!

**Last Updated**: October 2025
