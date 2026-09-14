# Settings Center

Open **File > Settings Center** to manage application behavior, global analysis defaults, and reusable analysis profiles in one place. The left navigation selects the broad area; the search box finds settings by name, category, or description.

## Save behavior

Settings Center stages changes while it is open.

- **Apply** validates and saves the current draft without closing the window.
- **OK** applies the draft, then closes the window.
- **Cancel** discards unapplied changes.
- **Reset Current Section** resets Program settings to their defaults, Global Analysis scalar fields to the values loaded when Settings Center opened, or the selected profile draft to its last saved values. It does not reset latency-window presets or M-wave recognition names; use their dedicated restore/reset controls instead.

Applying profile edits does not activate that profile in the main window. Use the main-window profile selector as the sole control for choosing the active profile.

## Program

Use **Program** for application-specific behavior that never belongs to an analysis profile:

- **Appearance**: window placement, interface scaling, fonts, and panel width.
- **Performance**: OpenGL, lazy raw-data opening, parallel loading, and cache warm-up.
- **Privacy and data**: restored selection, recent-file and path tracking, and clearing saved application data.

## Keyboard shortcuts

Use **Keyboard shortcuts** to customize main-window controls for rapid data curation and plotting. The shortcut column always shows the currently assigned keys, so it serves as the authoritative, preference-aware shortcut reference. Click a field, press the desired key combination, then choose **Apply** or **OK**. Changes take effect immediately after applying and are remembered for future launches. Clear a field to disable its command; shortcuts must be unique. **Restore Shortcut Defaults** returns every command to the bindings below. The session, dataset, experiment, and Plot tooltips update to show customized navigation and plotting shortcuts.

| Command | Default | Behavior |
| --- | --- | --- |
| Previous session / dataset / experiment | `Alt+Shift+1` / `Alt+Shift+2` / `Alt+Shift+3` | Select the preceding item at that level. |
| Next session / dataset / experiment | `Alt+1` / `Alt+2` / `Alt+3` | Select the following item at that level. |
| Mark session / dataset / experiment complete | `Ctrl+1` / `Ctrl+2` / `Ctrl+3` | Mark the currently selected item complete. |
| Mark session / dataset / experiment incomplete | `Ctrl+Shift+1` / `Ctrl+Shift+2` / `Ctrl+Shift+3` | Mark the currently selected item incomplete. |
| Plot | `Ctrl+P` | Run the selected plot without extracting data. |
| Plot and extract data | `Ctrl+Shift+P` | Run the selected plot and extract its data. |

The number consistently identifies the hierarchy: `1` is session, `2` is dataset, and `3` is experiment. Completion changes use MonStim's normal undoable command path, so **Edit > Undo** can revert them. Shortcuts are active only in the main window, not while Settings Center or another modal dialog is active.

## Global Analysis

Use **Global Analysis** for defaults shared across analyses. Its tabs contain plot appearance, signal processing, M-max estimation, imported-data defaults, and latency-window configuration.

Profiles inherit these values unless they explicitly override an eligible field. Some settings are intentionally global-only, including **M-wave Recognition Names**, because they control how MonStim identifies the M-response window for M-max calculations across the application.

The **Latency windows** tab contains two reusable global editors:

- **Latency Window Presets**: create and manage reusable named timing templates.
- **M-wave Recognition Names**: edit the recognized M-response aliases directly in the table. An empty list disables automatic M-wave recognition.

## Profiles

The **Profile Library** shows each profile and whether it is **Built-in** or **User**.

- Built-in profiles are read-only. Use **Duplicate** to create an editable user copy.
- User profiles can be created, duplicated, deleted, imported, and exported as YAML.
- When importing a profile with the same user-profile name, choose **Replace**, **Keep Both**, or **Cancel**.

Select a profile to open its editor. **Overview** shows the profile name, description, latency preset, and a comparison table for every explicit override: category, setting, profile value, and global default. Category tabs let you change eligible overrides; unchecked fields inherit the effective global value.

Profile files remain analysis-only overlays. They do not contain Program settings, and they do not replace session-level latency-window annotations already applied to data.

## Search

Enter a specific phrase such as `cache warm-up`, `axis label font size`, `profile import`, or `latency window`. Settings Center opens the matching area and filters unrelated tabs. Clear the search field to restore the full navigation.

## Related topics

- [Analysis profiles](analysis_profiles.md)
- [Configuration reference](../science/configuration_reference.md)
- [Latency windows](latency_windows.md)
- [Back to Help Library](index.md)
