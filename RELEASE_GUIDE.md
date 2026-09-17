# Release Preparation Guide

This guide outlines the steps needed to prepare a new release of MonStim Analyzer.

## Pre-Release Checklist

### 1. Update Version Numbers
- [ ] Update `monstim_gui/version.py` - set `VERSION = "X.X.X"`
- [ ] Update `QUICKSTART.md` - download links and executable names
- [ ] Update `docs/readme.md` - download links and executable names  
- [ ] Update `README.md`, `CITATION.cff`, and public beta/platform wording.
- [ ] Update GitHub issue templates in `.github/ISSUE_TEMPLATE/` - version placeholders
- [ ] Update `CHANGELOG.md` - add new version section with changes
- [ ] Update `monstim_gui\core\splash.py` if changing release type from alpha/beta to full release.

### 2. Pre-Build Steps
- [ ] Ensure all tests pass
- [ ] Verify application runs correctly with new version number

### 3. Build Process
```powershell
conda run -n monstim pyinstaller --clean win-main.spec
```

### 4. Post-Build Verification
- [ ] Test the built executable launches correctly
- [ ] Verify version number appears correctly in the application
- [ ] Check that all required files are included in the distribution
- [ ] Test basic functionality (import, plot, export)
- [ ] Verify `_internal\\MonStim Updater.exe`, `_internal\\monstim-release.json`, and `_internal\\LICENSE` are present in the release folder, and that neither updater nor manifest is at the release root.
- [ ] Test a staged update and rollback with a disposable per-user update root; confirm a sentinel experiment, settings file, and plug-in folder remain byte-for-byte unchanged.

### 5. Release Distribution
- [ ] Create GitHub release with appropriate tag (e.g., `v0.6.1`)
- [ ] Upload the distribution zip file
- [ ] Generate and upload `SHA256SUMS.txt` for every release artifact.
- [ ] Include changelog content in release notes
- [ ] Mark as pre-release if applicable
- [ ] Start notes from `.github/RELEASE_TEMPLATE.md`; state that the Windows release is unsigned.
- [ ] Create a **draft** GitHub Release with tag `vX.Y.Z`, upload the final ZIP, and do not replace that asset afterwards.
- [ ] Run `./tools/publish_release_catalog.ps1 -Version X.Y.Z -Archive "<final ZIP>"`. It verifies the packaged manifest/updater, calculates SHA-256, writes `SHA256SUMS.txt`, updates the source catalog, and signs `docs/updates.json` locally.
- [ ] Upload the generated `SHA256SUMS.txt` to the draft release.
- [ ] Review, commit, and push `tools/update_catalog.template.json` and `docs/updates.json`; wait for the Pages deployment to pass.
- [ ] Only then publish the GitHub Release and perform one clean-machine download/checksum/launch smoke test.

### Application update catalog

The application-update key is intentionally distinct from the importer-catalog key. Generate it once outside the repository, copy only the printed public key into `monstim_gui/updates.py`, and protect the private key:

```powershell
conda run -n monstim python -m tools.generate_update_catalog_key --private-key "$env:USERPROFILE\.monstim\update-catalog-ed25519.key"
```

The release wrapper above is the normal path. To sign a deliberately hand-edited catalog without committing the private key:

```powershell
conda run -n monstim python -m tools.sign_update_catalog `
  --catalog tools/update_catalog.template.json `
  --private-key "$env:USERPROFILE\.monstim\update-catalog-ed25519.key" `
  --output docs/updates.json
```

For a full explanation of the keys, recovery precautions, and official
importer-catalog workflow, see `docs/developer/releasing.md`.

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
- Updated importer compatibility and add-on catalog information when the plugin API changes.
