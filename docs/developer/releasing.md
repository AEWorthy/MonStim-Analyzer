# Publishing a signed MonStim release

This procedure is for authorized MonStim release maintainers. Contributors
without access to the project's signing keys can prepare and test release
artifacts, but should coordinate publication with a maintainer.

MonStim uses two Ed25519 signing keys to prove that a catalog entry was
approved by the project. They are **not** Windows code-signing certificates:
Windows may still show an unsigned-application warning.

Keep both private-key files outside the repository, backed up in an encrypted
location. The public key compiled into MonStim is safe to publish and is used
only to check signatures.

## Application releases

1. Build and test the exact Windows ZIP to distribute.
2. Create a **draft** GitHub Release tagged `vX.Y.Z`, then upload that exact
   ZIP. Do not replace the asset afterward.
3. From the repository root, run:

   ```powershell
   .\tools\publish_release_catalog.ps1 -Version 0.7.0 -Archive "dist\MonStim_Analyzer_v0.7.0-WIN.zip"
   ```

   The command validates the ZIP's `monstim-release.json` and updater helper,
   calculates SHA-256 itself, writes `SHA256SUMS.txt` beside the ZIP, replaces
   the matching source-catalog entry, and signs `docs/updates.json`. It does
   not send your private key anywhere.
4. Upload the generated `SHA256SUMS.txt` as a second draft-release asset.
5. Review the changed `tools/update_catalog.template.json` and
   `docs/updates.json`, commit and push them, and wait for the Pages workflow
   to succeed. The signed catalog then becomes available to MonStim.
6. Publish the GitHub Release. Confirm a clean machine can download the ZIP,
   verify its checksum, and launch it before announcing the release.

Use `-DryRun` first if desired. Use `-PrivateKey <path>` if the key is stored
somewhere other than `%USERPROFILE%\.monstim\update-catalog-ed25519.key`.

!!! warning
    A catalog entry can make the release visible to existing users. Never sign
    an archive that you have not personally tested, and never overwrite an
    already-published release asset. A checksum mismatch is intentionally
    treated as a failed update.

## Official importer add-ons

Only update the importer catalog when publishing or changing an official
add-on. Package and test the add-on ZIP, upload it to its GitHub Release, add
the reviewed identifier, version, compatibility range, asset URL, SHA-256, and
release notes URL to `tools/plugin_catalog.template.json`, then run:

```powershell
conda run -n monstim python -m tools.sign_plugin_catalog `
  --catalog tools/plugin_catalog.template.json `
  --private-key "$env:USERPROFILE\.monstim\plugin-catalog-ed25519.key" `
  --output docs/plugins/catalog.json
```

Commit/push both catalog files and wait for Pages to deploy. Do not re-sign the
plug-in catalog merely because the main application changed.

## Key rotation or suspected exposure

Do not delete, regenerate, or casually replace a key. Existing distributed
applications know only the currently embedded public key. If a private key is
lost or exposed, contact project/security support, generate a replacement, and
ship a new application version containing the new public key through the still
trusted channel. Treat a suspected exposure as urgent.
