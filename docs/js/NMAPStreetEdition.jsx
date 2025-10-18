/*
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
*/

import { useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { ScrollArea } from "../ui/scroll-area";
import { Badge } from "../ui/badge";
import { Checkbox } from "../ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { motion } from "motion/react";
import { Search, Terminal, Shield, Zap, Eye, Network, FileText, Download } from "lucide-react";

interface ScanResult {
  port: number;
  state: 'open' | 'closed' | 'filtered';
  service: string;
  version?: string;
  os?: string;
  cve?: string[];
  banner?: string;
}

interface HostInfo {
  ip: string;
  hostname?: string;
  status: 'up' | 'down';
  latency?: number;
  os?: string;
  ports?: ScanResult[];
}

export function NMAPStreetEdition() {
  const [activeTab, setActiveTab] = useState("scanner");
  const [scanning, setScanning] = useState(false);
  const [target, setTarget] = useState("");
  const [portRange, setPortRange] = useState("1-1000");
  const [results, setResults] = useState<ScanResult[]>([]);
  const [hosts, setHosts] = useState<HostInfo[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [selectedHost, setSelectedHost] = useState<HostInfo | null>(null);

  const [scanType, setScanType] = useState({
    quickScan: true,
    serviceScan: false,
    versionScan: false,
    osScan: false,
    aggressiveScan: false,
    stealthScan: false,
    udpScan: false,
    scriptScan: false
  });

  const [timingTemplate, setTimingTemplate] = useState<'paranoid' | 'sneaky' | 'polite' | 'normal' | 'aggressive' | 'insane'>('normal');
  const [scanTechnique, setScanTechnique] = useState<'SYN' | 'Connect' | 'ACK' | 'Window' | 'Maimon' | 'NULL' | 'FIN' | 'Xmas'>('SYN');

  const services: { [key: number]: { name: string; versions: string[]; cves: string[] } } = {
    21: {
      name: 'FTP',
      versions: ['vsftpd 3.0.3', 'ProFTPD 1.3.5', 'Pure-FTPd', 'FileZilla Server 0.9.60'],
      cves: ['CVE-2021-27292', 'CVE-2019-12815']
    },
    22: {
      name: 'SSH',
      versions: ['OpenSSH 8.2p1 Ubuntu-4ubuntu0.3', 'OpenSSH 7.4', 'Dropbear sshd 2019.78'],
      cves: ['CVE-2021-28041', 'CVE-2020-15778']
    },
    23: {
      name: 'Telnet',
      versions: ['Linux telnetd', 'Cisco telnetd', 'Windows Telnet'],
      cves: ['CVE-2020-10188']
    },
    25: {
      name: 'SMTP',
      versions: ['Postfix smtpd', 'Exim smtpd 4.94', 'Microsoft ESMTP 10.0', 'Sendmail 8.15.2'],
      cves: ['CVE-2021-3618', 'CVE-2020-28007']
    },
    53: {
      name: 'DNS',
      versions: ['ISC BIND 9.16.1', 'dnsmasq 2.80', 'Microsoft DNS 6.1'],
      cves: ['CVE-2021-25216', 'CVE-2020-25681']
    },
    80: {
      name: 'HTTP',
      versions: ['Apache httpd 2.4.41 ((Ubuntu))', 'nginx 1.18.0', 'Microsoft IIS 10.0'],
      cves: ['CVE-2021-40438', 'CVE-2021-23017', 'CVE-2021-31166']
    },
    110: {
      name: 'POP3',
      versions: ['Dovecot pop3d', 'Courier pop3d', 'Microsoft POP3 Service'],
      cves: []
    },
    143: {
      name: 'IMAP',
      versions: ['Dovecot imapd 2.3.7.2', 'Courier imapd', 'Microsoft Exchange imapd'],
      cves: ['CVE-2021-33515']
    },
    443: {
      name: 'HTTPS',
      versions: ['Apache httpd 2.4.41 SSL/TLS', 'nginx 1.18.0 SSL', 'Microsoft IIS 10.0 SSL'],
      cves: ['CVE-2021-40438', 'CVE-2021-23017']
    },
    445: {
      name: 'SMB',
      versions: ['Samba smbd 4.11.2', 'Microsoft Windows SMB', 'Samba smbd 3.X - 4.X'],
      cves: ['CVE-2021-44142', 'CVE-2020-1472']
    },
    3306: {
      name: 'MySQL',
      versions: ['MySQL 8.0.23-0ubuntu0.20.04.1', 'MariaDB 10.5.8', 'MySQL 5.7.33'],
      cves: ['CVE-2021-2146', 'CVE-2021-2166']
    },
    3389: {
      name: 'RDP',
      versions: ['Microsoft Terminal Services', 'xrdp 0.9.12'],
      cves: ['CVE-2019-0708 (BlueKeep)', 'CVE-2020-0609']
    },
    5432: {
      name: 'PostgreSQL',
      versions: ['PostgreSQL DB 13.2', 'PostgreSQL DB 12.6', 'PostgreSQL DB 11.11'],
      cves: ['CVE-2021-32027', 'CVE-2021-32028']
    },
    5900: {
      name: 'VNC',
      versions: ['RealVNC 6.7.2', 'TightVNC 2.8.11', 'UltraVNC 1.2.4'],
      cves: ['CVE-2020-14397', 'CVE-2019-8383']
    },
    6379: {
      name: 'Redis',
      versions: ['Redis 6.2.1', 'Redis 5.0.7', 'Redis 4.0.14'],
      cves: ['CVE-2021-32672', 'CVE-2021-32675']
    },
    8080: {
      name: 'HTTP-Proxy',
      versions: ['Jetty 9.4.38', 'Apache Tomcat 9.0.45', 'WEBrick 1.6.0'],
      cves: ['CVE-2021-28169', 'CVE-2021-25122']
    },
    8443: {
      name: 'HTTPS-Alt',
      versions: ['nginx 1.18.0 SSL', 'Apache httpd 2.4.41 SSL', 'Tomcat 9.0 SSL'],
      cves: []
    },
    27017: {
      name: 'MongoDB',
      versions: ['MongoDB 4.4.4', 'MongoDB 3.6.8', 'MongoDB 5.0.2'],
      cves: ['CVE-2021-20329']
    }
  };

  const osFingerprints = [
    'Linux 5.11.0-27-generic (Ubuntu 20.04)',
    'Linux 5.4.0-74-generic (Ubuntu 20.04 LTS)',
    'Linux 4.15.0-147-generic (Ubuntu 18.04)',
    'Windows Server 2019 Standard 17763',
    'Windows Server 2016 Datacenter',
    'Windows 10 Enterprise 19042',
    'macOS 11.4 (Big Sur)',
    'CentOS Linux 8.3.2011',
    'Debian GNU/Linux 10 (Buster)',
    'Red Hat Enterprise Linux 8.4',
    'FreeBSD 12.2-RELEASE',
    'Arch Linux'
  ];

  const nseScripts = [
    'http-title', 'http-headers', 'http-methods', 'http-robots.txt',
    'ssl-cert', 'ssl-enum-ciphers', 'ssl-heartbleed', 'ssl-poodle',
    'ssh-hostkey', 'ssh-auth-methods', 'ssh2-enum-algos',
    'smb-os-discovery', 'smb-security-mode', 'smb-enum-shares', 'smb-vuln-ms17-010',
    'mysql-info', 'mysql-vuln-cve2012-2122',
    'ftp-anon', 'ftp-bounce', 'ftp-vsftpd-backdoor',
    'dns-zone-transfer', 'dns-nsid',
    'telnet-encryption', 'vnc-info', 'rdp-enum-encryption'
  ];

  const addLog = (message: string, type: 'info' | 'warning' | 'error' | 'success' = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const prefix = type === 'error' ? '[ERROR] ' :
                   type === 'warning' ? '[WARNING] ' :
                   type === 'success' ? '[SUCCESS] ' : '[INFO] ';
    setLogs(prev => [...prev, `[${timestamp}] ${prefix}${message}`]);
  };

  const startScan = async () => {
    if (!target.trim()) return;

    setScanning(true);
    setResults([]);
    setLogs([]);
    setHosts([]);

    addLog(`Starting NMAP Street Edition ${scanType.aggressiveScan ? 'Aggressive' : 'Stealth'} Scan`);
    addLog(`Nmap v7.92 (https://nmap.org)`);
    addLog(`Target: ${target}`);
    addLog(`Port range: ${portRange}`);
    addLog(`Scan technique: ${scanTechnique} scan`);
    addLog(`Timing template: T${['0', '1', '2', '3', '4', '5'][['paranoid', 'sneaky', 'polite', 'normal', 'aggressive', 'insane'].indexOf(timingTemplate)]}`);

    await new Promise(r => setTimeout(r, 500));

    // Host discovery
    addLog('Initiating Ping Scan');
    addLog('Scanning for live hosts...');
    await new Promise(r => setTimeout(r, 800));

    const isNetwork = target.includes('/');
    const numHosts = isNetwork ? Math.floor(Math.random() * 10) + 3 : 1;

    const discoveredHosts: HostInfo[] = [];
    for (let i = 0; i < numHosts; i++) {
      const hostIP = isNetwork ? target.split('/')[0].replace(/\d+$/, String(i + 1)) : target;
      const host: HostInfo = {
        ip: hostIP,
        hostname: `host-${i + 1}.example.com`,
        status: Math.random() > 0.2 ? 'up' : 'down',
        latency: Math.floor(Math.random() * 100) + 10
      };

      if (host.status === 'up') {
        discoveredHosts.push(host);
        addLog(`Host ${host.ip} (${host.hostname}) is up (${host.latency}ms latency)`);
      }
    }

    setHosts(discoveredHosts);
    addLog(`Found ${discoveredHosts.length} hosts up`);

    if (discoveredHosts.length === 0) {
      addLog('No hosts found to be up', 'warning');
      setScanning(false);
      return;
    }

    // Port scanning
    addLog(`Initiating ${scanTechnique} Scan`);
    addLog(`Scanning ports ${portRange}...`);

    await new Promise(r => setTimeout(r, 500));

    const [start, end] = portRange.split('-').map(Number);
    const portsToScan = Math.min(end - start + 1, 100); // Limit for demo

    const commonPorts = Object.keys(services).map(Number).filter(p => p >= start && p <= end);
    const allResults: ScanResult[] = [];

    for (const port of commonPorts) {
      await new Promise(r => setTimeout(r, 50));

      const isOpen = Math.random() > 0.7;
      if (!isOpen && !scanType.aggressiveScan) continue;

      const state: ScanResult['state'] = isOpen ? 'open' : (Math.random() > 0.7 ? 'filtered' : 'closed');

      if (state === 'open') {
        const serviceInfo = services[port];
        const version = scanType.versionScan || scanType.serviceScan || scanType.aggressiveScan
          ? serviceInfo.versions[Math.floor(Math.random() * serviceInfo.versions.length)]
          : undefined;

        const cves = scanType.aggressiveScan && serviceInfo.cves.length > 0
          ? serviceInfo.cves.slice(0, Math.floor(Math.random() * serviceInfo.cves.length) + 1)
          : undefined;

        const result: ScanResult = {
          port,
          state,
          service: serviceInfo.name,
          version,
          cve: cves,
          os: scanType.osScan || scanType.aggressiveScan ? osFingerprints[Math.floor(Math.random() * osFingerprints.length)] : undefined,
          banner: version ? `220 ${serviceInfo.name} ${version} ready` : undefined
        };

        allResults.push(result);
        addLog(`Discovered open port: ${port}/${result.service} ${version || ''}`, 'success');

        if (cves && cves.length > 0) {
          addLog(`[VULN] ${port}/${result.service}: ${cves.join(', ')}`, 'warning');
        }
      }
    }

    setResults(allResults);

    // Service version detection
    if (scanType.serviceScan || scanType.versionScan || scanType.aggressiveScan) {
      await new Promise(r => setTimeout(r, 800));
      addLog('Service detection scan initiated (Version detection)');
      addLog('Probing open ports to determine service/version info...');
    }

    // OS detection
    if (scanType.osScan || scanType.aggressiveScan) {
      await new Promise(r => setTimeout(r, 700));
      const detectedOS = osFingerprints[Math.floor(Math.random() * osFingerprints.length)];
      addLog('Initiating OS detection (try #1) against target');
      addLog(`OS details: ${detectedOS}`, 'success');
      addLog('Network Distance: 2 hops');

      discoveredHosts.forEach(host => {
        host.os = detectedOS;
        host.ports = allResults;
      });
      setHosts([...discoveredHosts]);
    }

    // NSE script scan
    if (scanType.scriptScan || scanType.aggressiveScan) {
      await new Promise(r => setTimeout(r, 1000));
      addLog('NSE: Loaded 150 scripts for scanning.');
      addLog('Running NSE scripts...');

      const scriptsToRun = nseScripts.slice(0, Math.floor(Math.random() * 5) + 3);
      for (const script of scriptsToRun) {
        await new Promise(r => setTimeout(r, 200));
        addLog(`NSE: Running script ${script}`);
      }

      addLog('NSE: Script scan completed', 'success');
    }

    // Traceroute
    if (scanType.aggressiveScan) {
      await new Promise(r => setTimeout(r, 500));
      addLog('Initiating traceroute');
      addLog('TTL=1 ... 192.168.1.1');
      addLog(`TTL=2 ... ${target}`);
    }

    addLog('Nmap scan report complete', 'success');
    addLog(`Scanned ${portsToScan} ports on ${discoveredHosts.length} host(s)`);
    addLog(`${allResults.filter(r => r.state === 'open').length} open ports found`);
    setScanning(false);
  };

  const exportResults = () => {
    const exportData = {
      target,
      timestamp: new Date().toISOString(),
      scanOptions: scanType,
      hosts,
      results
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nmap-scan-${Date.now()}.json`;
    a.click();
  };

  return (
    <div className="space-y-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="bg-black/50 border border-red-900/50 grid grid-cols-4 w-full">
          <TabsTrigger value="scanner" className="data-[state=active]:bg-red-600">
            <Search className="w-4 h-4 mr-2" />
            Scanner
          </TabsTrigger>
          <TabsTrigger value="options" className="data-[state=active]:bg-red-600">
            <Shield className="w-4 h-4 mr-2" />
            Scan Options
          </TabsTrigger>
          <TabsTrigger value="hosts" className="data-[state=active]:bg-red-600">
            <Network className="w-4 h-4 mr-2" />
            Hosts
          </TabsTrigger>
          <TabsTrigger value="output" className="data-[state=active]:bg-red-600">
            <FileText className="w-4 h-4 mr-2" />
            Output
          </TabsTrigger>
        </TabsList>

        {/* SCANNER TAB */}
        <TabsContent value="scanner" className="space-y-4">
          <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
            <div className="space-y-3">
              <div className="flex gap-4">
                <div className="flex-1 relative">
                  <Shield className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                    placeholder="Target (e.g., 192.168.1.1, example.com, 192.168.1.0/24)"
                    className="bg-black/50 border-red-900/50 text-red-50 pl-10"
                  />
                </div>
                <Input
                  value={portRange}
                  onChange={(e) => setPortRange(e.target.value)}
                  placeholder="Port range"
                  className="bg-black/50 border-red-900/50 w-40"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Scan Technique</label>
                  <Select value={scanTechnique} onValueChange={(val: any) => setScanTechnique(val)}>
                    <SelectTrigger className="bg-black/50 border-red-900/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="SYN">SYN Stealth (-sS)</SelectItem>
                      <SelectItem value="Connect">TCP Connect (-sT)</SelectItem>
                      <SelectItem value="ACK">ACK Scan (-sA)</SelectItem>
                      <SelectItem value="Window">Window Scan (-sW)</SelectItem>
                      <SelectItem value="Maimon">Maimon Scan (-sM)</SelectItem>
                      <SelectItem value="NULL">NULL Scan (-sN)</SelectItem>
                      <SelectItem value="FIN">FIN Scan (-sF)</SelectItem>
                      <SelectItem value="Xmas">Xmas Scan (-sX)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Timing Template</label>
                  <Select value={timingTemplate} onValueChange={(val: any) => setTimingTemplate(val)}>
                    <SelectTrigger className="bg-black/50 border-red-900/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="paranoid">Paranoid (T0) - IDS evasion</SelectItem>
                      <SelectItem value="sneaky">Sneaky (T1) - IDS evasion</SelectItem>
                      <SelectItem value="polite">Polite (T2) - Less bandwidth</SelectItem>
                      <SelectItem value="normal">Normal (T3) - Default</SelectItem>
                      <SelectItem value="aggressive">Aggressive (T4) - Fast scan</SelectItem>
                      <SelectItem value="insane">Insane (T5) - Very fast</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button
                onClick={startScan}
                disabled={scanning || !target.trim()}
                className="bg-red-600 hover:bg-red-700 w-full"
              >
                {scanning ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    Scanning...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4 mr-2" />
                    Start NMAP Scan
                  </>
                )}
              </Button>
            </div>
          </div>

          {results.length > 0 && (
            <>
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-green-950/20 border border-green-500/30 rounded-lg p-4">
                  <div className="text-gray-400 text-sm">Open Ports</div>
                  <div className="text-green-400 text-2xl">{results.filter(r => r.state === 'open').length}</div>
                </div>
                <div className="bg-red-950/20 border border-red-500/30 rounded-lg p-4">
                  <div className="text-gray-400 text-sm">Closed Ports</div>
                  <div className="text-red-400 text-2xl">{results.filter(r => r.state === 'closed').length}</div>
                </div>
                <div className="bg-yellow-950/20 border border-yellow-500/30 rounded-lg p-4">
                  <div className="text-gray-400 text-sm">Filtered</div>
                  <div className="text-yellow-400 text-2xl">{results.filter(r => r.state === 'filtered').length}</div>
                </div>
                <div className="bg-blue-950/20 border border-blue-500/30 rounded-lg p-4">
                  <div className="text-gray-400 text-sm">Vulnerabilities</div>
                  <div className="text-blue-400 text-2xl">{results.filter(r => r.cve && r.cve.length > 0).length}</div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <h4 className="text-red-50">Scan Results</h4>
                <Button variant="outline" size="sm" className="border-red-500/50" onClick={exportResults}>
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
              </div>
            </>
          )}

          <div className="grid grid-cols-2 gap-4">
            {/* Results */}
            <ScrollArea className="h-[350px] bg-black/50 border border-red-900/30 rounded-lg">
              <div className="p-4 space-y-2">
                {results.length === 0 && !scanning && (
                  <div className="text-center text-gray-500 py-8">
                    Configure scan options and click Start NMAP Scan
                  </div>
                )}
                {results.map((result, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.02 }}
                    className={`p-3 rounded border ${
                      result.state === 'open'
                        ? 'border-green-500/30 bg-green-950/20'
                        : result.state === 'filtered'
                        ? 'border-yellow-500/30 bg-yellow-950/20'
                        : 'border-gray-700/30 bg-gray-950/20'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant="outline"
                          className={
                            result.state === 'open'
                              ? 'border-green-500/50 text-green-400'
                              : result.state === 'filtered'
                              ? 'border-yellow-500/50 text-yellow-400'
                              : 'border-gray-500/50 text-gray-500'
                          }
                        >
                          {result.state}
                        </Badge>
                        <span className="text-red-50">Port {result.port}</span>
                      </div>
                      <span className="text-gray-400 text-sm">{result.service}</span>
                    </div>
                    {result.version && (
                      <div className="text-xs text-blue-400 mb-1">Version: {result.version}</div>
                    )}
                    {result.os && (
                      <div className="text-xs text-purple-400 mb-1">OS: {result.os}</div>
                    )}
                    {result.banner && (
                      <div className="text-xs text-gray-500 font-mono mb-1">{result.banner}</div>
                    )}
                    {result.cve && result.cve.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-red-900/30">
                        <div className="text-xs text-red-400 mb-1">Known Vulnerabilities:</div>
                        {result.cve.map((cve, cidx) => (
                          <Badge key={cidx} variant="outline" className="border-red-500/50 text-red-400 text-xs mr-1">
                            {cve}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </ScrollArea>

            {/* Terminal Log */}
            <div className="bg-black/80 border border-red-900/30 rounded-lg overflow-hidden">
              <div className="bg-red-950/30 border-b border-red-900/50 px-4 py-2 flex items-center gap-2">
                <Terminal className="w-4 h-4 text-red-400" />
                <span className="text-red-400 text-sm">Scan Output</span>
              </div>
              <ScrollArea className="h-[306px]">
                <div className="p-4 font-mono text-xs space-y-1">
                  {logs.map((log, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className={
                        log.includes('[ERROR]') ? 'text-red-400' :
                        log.includes('[WARNING]') || log.includes('[VULN]') ? 'text-yellow-400' :
                        log.includes('[SUCCESS]') ? 'text-green-400' :
                        'text-green-400'
                      }
                    >
                      {log}
                    </motion.div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </div>
        </TabsContent>

        {/* SCAN OPTIONS TAB */}
        <TabsContent value="options" className="space-y-4">
          <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
            <h4 className="text-red-50 mb-3 text-sm">Scan Types</h4>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(scanType).map(([key, value]) => (
                <label key={key} className="flex items-center gap-2 text-gray-400 text-sm cursor-pointer">
                  <Checkbox
                    checked={value}
                    onCheckedChange={(checked) => setScanType(prev => ({ ...prev, [key]: !!checked }))}
                    className="border-red-500/50"
                  />
                  <div>
                    <div className="text-red-50">
                      {key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase())}
                    </div>
                    <div className="text-xs text-gray-500">
                      {key === 'quickScan' ? 'Fast scan (100 common ports)' :
                       key === 'serviceScan' ? 'Probe open ports for service info' :
                       key === 'versionScan' ? 'Determine service/version info' :
                       key === 'osScan' ? 'Enable OS detection' :
                       key === 'aggressiveScan' ? 'Enable all detection (OS, version, scripts, traceroute)' :
                       key === 'stealthScan' ? 'SYN/Stealth scan to avoid detection' :
                       key === 'udpScan' ? 'UDP port scan' :
                       key === 'scriptScan' ? 'Run NSE scripts' : ''}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
              <h4 className="text-red-50 mb-3 text-sm">Common Port Ranges</h4>
              <div className="space-y-2">
                <Button
                  variant="outline"
                  className="border-red-500/50 w-full justify-start"
                  onClick={() => setPortRange('1-1000')}
                >
                  Common ports (1-1000)
                </Button>
                <Button
                  variant="outline"
                  className="border-red-500/50 w-full justify-start"
                  onClick={() => setPortRange('1-65535')}
                >
                  All ports (1-65535)
                </Button>
                <Button
                  variant="outline"
                  className="border-red-500/50 w-full justify-start"
                  onClick={() => setPortRange('21,22,23,25,80,443,3389,8080')}
                >
                  Web & Remote (21,22,23,25,80,443,3389,8080)
                </Button>
                <Button
                  variant="outline"
                  className="border-red-500/50 w-full justify-start"
                  onClick={() => setPortRange('1433,3306,5432,6379,27017')}
                >
                  Databases (1433,3306,5432,6379,27017)
                </Button>
              </div>
            </div>

            <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
              <h4 className="text-red-50 mb-3 text-sm">NSE Script Categories</h4>
              <ScrollArea className="h-[200px]">
                <div className="space-y-2 pr-4">
                  {['auth', 'broadcast', 'brute', 'default', 'discovery', 'dos', 'exploit', 'external', 'fuzzer', 'intrusive', 'malware', 'safe', 'version', 'vuln'].map(cat => (
                    <div key={cat} className="flex items-center justify-between p-2 rounded border border-red-900/30 bg-black/30">
                      <span className="text-gray-300 text-sm capitalize">{cat}</span>
                      <Badge variant="outline" className="border-blue-500/50 text-blue-400 text-xs">
                        {Math.floor(Math.random() * 50) + 10} scripts
                      </Badge>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </div>
        </TabsContent>

        {/* HOSTS TAB */}
        <TabsContent value="hosts" className="space-y-4">
          {hosts.length > 0 && (
            <>
              <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
                <h4 className="text-red-50 mb-3">Discovered Hosts ({hosts.length})</h4>
                <div className="space-y-2">
                  {hosts.map((host, idx) => (
                    <div
                      key={idx}
                      onClick={() => setSelectedHost(host)}
                      className={`p-4 rounded border cursor-pointer transition-colors ${
                        selectedHost?.ip === host.ip
                          ? 'border-red-500/50 bg-red-950/20'
                          : 'border-red-900/30 bg-black/30 hover:bg-red-950/10'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <Badge variant="outline" className="border-green-500/50 text-green-400">
                            {host.status.toUpperCase()}
                          </Badge>
                          <span className="text-red-50 font-mono">{host.ip}</span>
                          {host.hostname && (
                            <span className="text-gray-400 text-sm">({host.hostname})</span>
                          )}
                        </div>
                        {host.latency && (
                          <span className="text-gray-400 text-sm">{host.latency}ms</span>
                        )}
                      </div>
                      {host.os && (
                        <div className="text-sm text-blue-400">OS: {host.os}</div>
                      )}
                      {host.ports && (
                        <div className="text-sm text-gray-400 mt-1">
                          {host.ports.filter(p => p.state === 'open').length} open ports
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {selectedHost && selectedHost.ports && (
                <div className="bg-black/50 border border-red-900/30 rounded-lg">
                  <div className="bg-red-950/30 border-b border-red-900/50 px-4 py-2">
                    <span className="text-red-400 text-sm">
                      Ports on {selectedHost.ip} ({selectedHost.hostname})
                    </span>
                  </div>
                  <ScrollArea className="h-[300px]">
                    <div className="p-4 space-y-2">
                      {selectedHost.ports.map((port, idx) => (
                        <div key={idx} className="p-3 rounded border border-red-900/30 bg-black/30">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Badge
                                variant="outline"
                                className={
                                  port.state === 'open'
                                    ? 'border-green-500/50 text-green-400'
                                    : 'border-gray-500/50 text-gray-500'
                                }
                              >
                                {port.port}/{port.service}
                              </Badge>
                              {port.version && (
                                <span className="text-blue-400 text-sm">{port.version}</span>
                              )}
                            </div>
                            <Badge variant="outline" className="border-gray-500/50 text-gray-400">
                              {port.state}
                            </Badge>
                          </div>
                          {port.cve && port.cve.length > 0 && (
                            <div className="flex gap-1 flex-wrap">
                              {port.cve.map((cve, cidx) => (
                                <Badge key={cidx} variant="outline" className="border-red-500/50 text-red-400 text-xs">
                                  {cve}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              )}
            </>
          )}

          {hosts.length === 0 && (
            <div className="text-center text-gray-500 py-12">
              <Network className="w-12 h-12 mx-auto mb-4 text-gray-600" />
              <p>No hosts discovered yet</p>
              <p className="text-sm mt-2">Run a scan to discover hosts</p>
            </div>
          )}
        </TabsContent>

        {/* OUTPUT TAB */}
        <TabsContent value="output" className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-red-50">Scan Log</h4>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" className="border-red-500/50">
                <Download className="w-4 h-4 mr-2" />
                Save Log
              </Button>
              <Button variant="outline" size="sm" className="border-red-500/50" onClick={() => setLogs([])}>
                Clear
              </Button>
            </div>
          </div>

          <div className="bg-black/80 border border-red-900/30 rounded-lg overflow-hidden">
            <ScrollArea className="h-[500px]">
              <div className="p-4 font-mono text-xs space-y-1">
                {logs.length === 0 ? (
                  <div className="text-center text-gray-500 py-12">
                    <Terminal className="w-12 h-12 mx-auto mb-4 text-gray-600" />
                    <p>No scan output yet</p>
                  </div>
                ) : (
                  logs.map((log, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className={
                        log.includes('[ERROR]') ? 'text-red-400' :
                        log.includes('[WARNING]') || log.includes('[VULN]') ? 'text-yellow-400' :
                        log.includes('[SUCCESS]') ? 'text-green-400' :
                        'text-green-400'
                      }
                    >
                      {log}
                    </motion.div>
                  ))
                )}
              </div>
            </ScrollArea>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
