# UI scaling and display troubleshooting

## Purpose

Use this guide when text, tables, or dialog controls are clipped, unusually small, or appear on the wrong monitor. Display scaling is controlled by the operating system and Qt; the analysis results are not changed by display scaling.

## Steps: first checks

1. Maximize the window, then restore it if only one dialog is clipped.
2. Confirm that Windows **Settings > System > Display > Scale** uses a recommended value for the monitor currently showing MonStim.
3. If using multiple monitors with different scale factors, move the main window to the intended monitor before opening a dialog again.
4. Close and reopen MonStim after changing Windows display scale, resolution, or monitor arrangement.

## If a saved window position is off-screen

Disconnecting a monitor can leave a saved window position outside the visible desktop. Reconnect the monitor or use the operating system's window-management shortcut to move the window back onto a visible display. If the problem persists, contact the maintainer with the display details above; resetting saved UI settings is an advanced maintenance action.

## When to report a display issue

Include the MonStim version, operating system version, monitor resolutions and scale factors, whether monitors were connected or disconnected while MonStim was open, and a screenshot showing the affected dialog. This lets support distinguish a layout issue from a display-scaling or saved-state issue.

## Related topics

- [Troubleshooting](troubleshooting.md)
- [Back to Help Library](index.md)
