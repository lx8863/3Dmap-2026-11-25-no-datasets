using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Forms;

internal static class Launcher
{
    internal static readonly string DataFolder =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "3Dmap");
    internal static readonly string AddressFile = Path.Combine(DataFolder, "active-url.txt");
    private static readonly Regex LocalAddress = new Regex(@"^http://127\.0\.0\.1:\d{1,5}/$");

    [STAThread]
    private static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        try
        {
            Directory.CreateDirectory(DataFolder);
            bool first;
            using (var single = new Mutex(true, @"Local\3DmapPortable", out first))
            {
                if (!first)
                {
                    if (File.Exists(AddressFile))
                    {
                        Open(File.ReadAllText(AddressFile).Trim());
                    }
                    else
                    {
                        MessageBox.Show("3Dmap 正在启动。", "3Dmap", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    return;
                }
                File.Delete(AddressFile);
                using (var studio = new StudioProcess(AppDomain.CurrentDomain.BaseDirectory))
                {
                    Application.Run(studio);
                }
            }
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "3Dmap 启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    internal static bool IsAddress(string value)
    {
        return value != null && LocalAddress.IsMatch(value);
    }

    internal static void Open(string address)
    {
        if (!IsAddress(address)) return;
        string edge = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
                                   "Microsoft", "Edge", "Application", "msedge.exe");
        if (File.Exists(edge))
        {
            Process.Start(new ProcessStartInfo(edge, "--app=" + address) { UseShellExecute = false });
        }
        else
        {
            Process.Start(new ProcessStartInfo(address) { UseShellExecute = true });
        }
    }
}

internal sealed class StudioProcess : ApplicationContext
{
    private const string ReadyPrefix = "COLMAP workbench: ";
    private readonly WindowsFormsSynchronizationContext ui = new WindowsFormsSynchronizationContext();
    private readonly object logLock = new object();
    private readonly string logPath = Path.Combine(Launcher.DataFolder, "3Dmap.log");
    private readonly Process server;
    private readonly NotifyIcon tray;
    private string address;
    private bool closing;

    internal StudioProcess(string folder)
    {
        string python = Path.Combine(folder, "runtime", "python.exe");
        if (!File.Exists(python))
            throw new FileNotFoundException("运行文件缺失。请重新复制完整的 3Dmap 文件夹。");

        File.WriteAllText(logPath, "");
        string data = Path.Combine(Launcher.DataFolder, "projects");
        var start = new ProcessStartInfo(python)
        {
            Arguments = "-B -u \"web\\serve.py\" --port 0 --desktop --data-root \"" + data +
                        "\" --vggt-python \"" + python + "\"",
            WorkingDirectory = folder,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
        start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        server = new Process { StartInfo = start, EnableRaisingEvents = true };
        server.OutputDataReceived += OnOutput;
        server.ErrorDataReceived += (sender, args) => Log(args.Data);
        server.Exited += (sender, args) => ui.Post(delegate
        {
            if (closing) return;
            MessageBox.Show("请查看运行日志\n" + logPath, "3Dmap 服务已停止",
                            MessageBoxButtons.OK, MessageBoxIcon.Warning);
            ExitThread();
        }, null);
        server.Start();
        server.BeginOutputReadLine();
        server.BeginErrorReadLine();

        var menu = new ContextMenuStrip();
        menu.Items.Add("打开工作台", null, (sender, args) => Launcher.Open(address));
        menu.Items.Add("退出 3Dmap", null, (sender, args) => ExitThread());
        tray = new NotifyIcon
        {
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath),
            Text = "3Dmap",
            ContextMenuStrip = menu,
            Visible = true
        };
        tray.DoubleClick += (sender, args) => Launcher.Open(address);
    }

    private void OnOutput(object sender, DataReceivedEventArgs args)
    {
        string line = args.Data;
        Log(line);
        if (line == null || !line.StartsWith(ReadyPrefix, StringComparison.Ordinal)) return;
        string candidate = line.Substring(ReadyPrefix.Length);
        if (!Launcher.IsAddress(candidate)) return;
        ui.Post(delegate
        {
            if (closing || address != null) return;
            address = candidate;
            File.WriteAllText(Launcher.AddressFile, address);
            Launcher.Open(address);
        }, null);
    }

    private void Log(string line)
    {
        if (line == null) return;
        lock (logLock)
        {
            try { File.AppendAllText(logPath, line + Environment.NewLine); }
            catch (IOException) { }
        }
    }

    protected override void ExitThreadCore()
    {
        if (closing) return;
        closing = true;
        tray.Visible = false;
        tray.ContextMenuStrip.Dispose();
        tray.Icon.Dispose();
        tray.Dispose();
        File.Delete(Launcher.AddressFile);
        try
        {
            server.StandardInput.Close();
            if (!server.WaitForExit(12000)) server.Kill();
        }
        finally
        {
            server.Dispose();
            ui.Dispose();
            base.ExitThreadCore();
        }
    }
}
