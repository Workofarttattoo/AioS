// Electron wrapper for Ai|oS Desktop Launcher
// Run with: electron launcher_electron.js

const { app, BrowserWindow, ipcMain } = require('electron');
const { exec } = require('child_process');
const path = require('path');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        fullscreen: true,
        frame: false,
        backgroundColor: '#000000',
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    mainWindow.loadFile('aios_launcher.html');

    // Handle tool launches
    ipcMain.on('launch-tool', (event, toolName) => {
        const command = `python -m tools.${toolName} --gui`;
        console.log(`Launching: ${command}`);

        exec(command, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error launching ${toolName}:`, error);
                event.reply('tool-launch-result', {
                    tool: toolName,
                    success: false,
                    error: error.message
                });
                return;
            }

            event.reply('tool-launch-result', {
                tool: toolName,
                success: true,
                output: stdout
            });
        });
    });

    // Exit fullscreen with Escape
    mainWindow.webContents.on('before-input-event', (event, input) => {
        if (input.key === 'Escape') {
            app.quit();
        }
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    app.quit();
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
