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
