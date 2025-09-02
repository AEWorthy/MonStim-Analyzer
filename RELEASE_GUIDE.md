# Release Preparation Guide

This guide outlines the steps needed to prepare a new release of MonStim Analyzer.

## Pre-Release Checklist

### 1. Update Version Numbers
- [ ] Update `monstim_gui/version.py` - set `VERSION = "X.X.X"`
- [ ] Update `QUICKSTART.md` - download links and executable names
- [ ] Update `docs/readme.md` - download links and executable names  
- [ ] Update GitHub issue templates in `.github/ISSUE_TEMPLATE/` - version placeholders
- [ ] Update `CHANGELOG.md` - add new version section with changes
- [ ] Update `monstim_gui\core\splash.py` if changing release type from alpha/beta to full release.

### 2. Pre-Build Steps
- [ ] Ensure all tests pass
- [ ] Verify application runs correctly with new version number

### 3. Build Process
```powershell
# Clean previous builds
pyinstaller --clean win-main.spec
```

### 4. Post-Build Verification
- [ ] Test the built executable launches correctly
- [ ] Verify version number appears correctly in the application
- [ ] Check that all required files are included in the distribution
- [ ] Test basic functionality (import, plot, export)

### 5. Release Distribution
- [ ] Create GitHub release with appropriate tag (e.g., `v0.4.2`)
- [ ] Upload the distribution zip file
- [ ] Include changelog content in release notes
- [ ] Mark as pre-release if applicable

## Build Configuration Notes

The `win-main.spec` file contains debug/release toggles:

**For Release:**
- `noarchive=False`
- `optimize=1` 
- `debug=False`
- `console=False`
- `disable_windowed_traceback=True`

**For Debug:**
- `noarchive=True`
- `optimize=0`
- `debug=True` 
- `console=True`
- `disable_windowed_traceback=False`

## Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in backward-compatible manner  
- **PATCH**: Backward-compatible bug fixes

## Documentation Updates

When releasing, ensure all documentation reflects:
- New features and improvements
- Updated installation instructions
- Any breaking changes or migration notes
- Updated screenshots if UI has changed significantly
