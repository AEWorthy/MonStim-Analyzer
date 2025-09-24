# Contributing to MonStim Analyzer

Thank you for your interest in contributing to MonStim Analyzer! This project aims to provide researchers with powerful, user-friendly tools for EMG data analysis and visualization.

## Quick Start

### For Bug Reports and Feature Requests
- Use our [issue templates](https://github.com/AEWorthy/MonStim-Analyzer/issues/new/choose) to report bugs or suggest features
- Search existing issues first to avoid duplicates
- Provide as much detail as possible, including system information and steps to reproduce

### For Code Contributions
1. **Fork the repository** and create a feature branch
2. **Set up your development environment** (see Development Setup below)
3. **Make your changes** following our coding standards
4. **Test thoroughly** on different systems and with various data
5. **Submit a pull request** using our PR template

## Development Setup

### Prerequisites
- Python 3.8 or later
- Git for version control
- PyQt6 and scientific computing libraries

### Installation Steps

1. **Clone your fork:**
   ```bash
   git clone https://github.com/your-username/MonStim-Analyzer.git
   cd MonStim-Analyzer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test your setup:**
   ```bash
   python main.py
   ```

## How to Contribute

### Reporting Bugs

Use our [Bug Report template](https://github.com/AEWorthy/MonStim-Analyzer/issues/new?template=bug_report.yml) and include:

- **System information:** OS, Python version, MonStim Analyzer version
- **Detailed steps to reproduce** the issue
- **Expected vs. actual behavior**
- **Error messages or screenshots**
- **Sample data** (if safe to share and relevant)

### Suggesting Features

Use our [Feature Request template](https://github.com/AEWorthy/MonStim-Analyzer/issues/new?template=feature_request.yml) and describe:

- **The problem** your feature would solve
- **Your proposed solution** in detail
- **Use cases** where this would be helpful
- **Alternative approaches** you've considered

### Improving Documentation

Documentation improvements are always welcome! This includes:

- **README updates** for new features
- **User guide improvements** for clarity
- **Code documentation** and docstrings
- **Example scripts** and tutorials

Use our [Documentation template](https://github.com/AEWorthy/MonStim-Analyzer/issues/new?template=documentation.yml) for documentation issues.

### Code Contributions

We welcome code contributions that:

- **Fix bugs** reported in our issue tracker
- **Implement requested features** with community discussion
- **Improve performance** for large datasets
- **Enhance usability** of the interface
- **Add tests** to improve code coverage

## Development Guidelines

### Code Style

- **Follow PEP 8** for Python code style
- **Use descriptive variable names** and function names
- **Add docstrings** to all public functions and classes
- **Keep functions focused** on a single responsibility
- **Comment complex algorithms** or business logic

### UI/UX Guidelines

- **Test on different screen sizes** and DPI settings
- **Ensure accessibility** with appropriate font sizes and contrast
- **Provide clear error messages** and user feedback
- **Follow existing UI patterns** for consistency
- **Test responsive design** on various display configurations

### Testing Requirements

- **Automated tests** with pytest (see [Testing Guide](docs/testing.md))
- **Manual testing** on multiple operating systems
- **Test with various data sizes** (small and large datasets)
- **Verify UI scaling** on different DPI settings
- **Test error handling** with invalid inputs
- **Ensure backward compatibility** with existing data

### Performance Considerations

- **Profile performance** with large datasets
- **Optimize import operations** for large files
- **Consider memory usage** with multiple experiments
- **Use background processing** for long operations
- **Provide progress feedback** to users

## Project Structure

```
MonStim-Analyzer/
├── main.py                     # Application entry point
├── monstim_gui/               # GUI application code
│   ├── core/                  # Core application logic
│   ├── dialogs/               # Dialog windows
│   ├── widgets/               # Custom UI components
│   └── plotting/              # Plotting functionality
├── monstim_signals/           # Signal processing and analysis
│   ├── core/                  # Core data structures
│   ├── domain/                # Business logic
│   ├── io/                    # Data import/export
│   ├── transform/             # Signal processing
│   └── plotting/              # Plotting backends
├── docs/                      # Documentation
├── data/                      # Sample data (ignored in git)
└── tests/                     # Test files (future)
```

## Testing

### Manual Testing Checklist

Before submitting a PR, please test:

- [ ] **Basic functionality:** Import, analyze, and plot data
- [ ] **Different data sizes:** Small and large datasets
- [ ] **Multiple operating systems:** Windows (primary), macOS, Linux
- [ ] **UI scaling:** Different DPI settings and screen resolutions
- [ ] **Error handling:** Invalid files, corrupted data, missing files
- [ ] **Performance:** Reasonable response times with typical datasets

### Test Data

- Do NOT use the `data/` directory for tests.
- Use curated 'golden' fixtures under `tests/fixtures/golden/` directory for tests and import them into temporary directories during testing.
- Negative cases live under `tests/fixtures/golden/invalid/` for malformed inputs and wrong naming.
- A dedicated test (`tests/test_golden_channel_counts.py`) enforces a ≥ 2 channel policy for all golden imports.

## Areas for Contribution

### High Priority
- **Performance optimization** for large datasets
- **Cross-platform compatibility** improvements
- **Error handling** and user feedback
- **Accessibility** improvements
- **Test suite** development

### Medium Priority
- **Additional plot types** for specific analysis needs
- **Export format** extensions
- **Data validation** improvements
- **UI/UX** enhancements
- **Documentation** expansions

### Nice to Have
- **Plugin system** for custom analyses
- **Batch processing** improvements
- **Integration** with other tools
- **Internationalization** support

## Getting Help

- **GitHub Issues:** For bugs, features, and technical questions
- **GitHub Discussions:** For general questions and community discussion
- **Email:** aeworthy@emory.edu for sensitive issues or collaboration inquiries
- **Documentation:** Check the [comprehensive README](docs/readme.md) first

## License

By contributing to MonStim Analyzer, you agree that your contributions will be licensed under the same [BSD 2-Clause License](LICENSE) that covers the project.

---

**Thank you for helping improving MonStim Analyzer!**
