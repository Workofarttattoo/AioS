/*
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
*/

import { useState, useEffect } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { ScrollArea } from "../ui/scroll-area";
import { Badge } from "../ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Checkbox } from "../ui/checkbox";
import { motion } from "motion/react";
import { Lock, Key, Zap, Database, Shield, Cpu, Download, Upload, Search } from "lucide-react";

interface HashResult {
  hash: string;
  algorithm: string;
  plaintext?: string;
  cracked: boolean;
  attempts: number;
  timeElapsed: number;
}

interface DictionaryStats {
  loaded: number;
  total: number;
  progress: number;
}

export function HashSolver() {
  const [activeTab, setActiveTab] = useState("single");
  const [running, setRunning] = useState(false);

  // Single hash mode
  const [singleHash, setSingleHash] = useState("");
  const [detectedAlgorithm, setDetectedAlgorithm] = useState<string>("");
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("auto");
  const [result, setResult] = useState<HashResult | null>(null);

  // Batch mode
  const [batchHashes, setBatchHashes] = useState("");
  const [batchResults, setBatchResults] = useState<HashResult[]>([]);

  // Attack configuration
  const [attackMode, setAttackMode] = useState<'dictionary' | 'brute' | 'hybrid' | 'rule' | 'mask'>('dictionary');
  const [dictionary, setDictionary] = useState("rockyou.txt");
  const [dictionaryStats, setDictionaryStats] = useState<DictionaryStats | null>(null);
  const [rules, setRules] = useState<string[]>([]);

  // Brute force settings
  const [charset, setCharset] = useState("abcdefghijklmnopqrstuvwxyz0123456789");
  const [minLength, setMinLength] = useState(1);
  const [maxLength, setMaxLength] = useState(8);

  // Mask attack
  const [maskPattern, setMaskPattern] = useState("?l?l?l?l?d?d?d?d");

  // Performance
  const [threads, setThreads] = useState(4);
  const [hashRate, setHashRate] = useState(0);
  const [progress, setProgress] = useState(0);
  const [eta, setEta] = useState("--:--:--");

  const algorithms = [
    { name: 'MD5', pattern: /^[a-f0-9]{32}$/i, id: 'md5' },
    { name: 'SHA1', pattern: /^[a-f0-9]{40}$/i, id: 'sha1' },
    { name: 'SHA256', pattern: /^[a-f0-9]{64}$/i, id: 'sha256' },
    { name: 'SHA512', pattern: /^[a-f0-9]{128}$/i, id: 'sha512' },
    { name: 'NTLM', pattern: /^[a-f0-9]{32}$/i, id: 'ntlm' },
    { name: 'bcrypt', pattern: /^\$2[ayb]\$.{56}$/i, id: 'bcrypt' },
    { name: 'MD5(Unix)', pattern: /^\$1\$[a-zA-Z0-9./]{8}\$[a-zA-Z0-9./]{22}$/i, id: 'md5crypt' },
    { name: 'SHA512(Unix)', pattern: /^\$6\$[a-zA-Z0-9./]{16}\$[a-zA-Z0-9./]{86}$/i, id: 'sha512crypt' },
    { name: 'MySQL', pattern: /^[a-f0-9]{40}$/i, id: 'mysql' },
    { name: 'PostgreSQL', pattern: /^md5[a-f0-9]{32}$/i, id: 'postgres' },
    { name: 'LM', pattern: /^[a-f0-9]{32}$/i, id: 'lm' },
    { name: 'phpass', pattern: /^\$P\$[a-zA-Z0-9./]{31}$/i, id: 'phpass' }
  ];

  const commonDictionaries = [
    { name: 'rockyou.txt', size: '14.3M', words: '14344391' },
    { name: 'Top 10k Passwords', size: '87K', words: '10000' },
    { name: 'Common Passwords', size: '245K', words: '62000' },
    { name: 'English Wordlist', size: '4.8M', words: '479828' },
    { name: 'Custom Dictionary', size: '0', words: '0' }
  ];

  const rulesList = [
    'None',
    'best64.rule',
    'toggles.rule',
    'leetspeak.rule',
    'append_numbers.rule',
    'prepend_symbols.rule',
    'reverse.rule',
    'duplicate.rule'
  ];

  const knownHashes: {[key: string]: string} = {
    // MD5
    '5f4dcc3b5aa765d61d8327deb882cf99': 'password',
    '098f6bcd4621d373cade4e832627b4f6': 'test',
    '827ccb0eea8a706c4c34a16891f84e7b': '12345',
    'e10adc3949ba59abbe56e057f20f883e': '123456',
    '25d55ad283aa400af464c76d713c07ad': '12345678',
    // SHA1
    '5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8': 'password',
    'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3': 'test',
    // SHA256
    '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8': 'password',
    '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08': 'test',
    // NTLM
    '8846f7eaee8fb117ad06bdd830b7586c': 'password',
    '0cb6948805f797bf2a82807973b89537': 'admin',
  };

  // Auto-detect hash algorithm
  useEffect(() => {
    if (singleHash.trim()) {
      const detected = algorithms.find(alg => alg.pattern.test(singleHash.trim()));
      if (detected) {
        setDetectedAlgorithm(detected.name);
      } else {
        setDetectedAlgorithm("Unknown");
      }
    } else {
      setDetectedAlgorithm("");
    }
  }, [singleHash]);

  // Simulate hash cracking
  const crackHash = async (hash: string, isBatch: boolean = false): Promise<HashResult> => {
    const startTime = Date.now();
    let attempts = 0;
    let cracked = false;
    let plaintext: string | undefined;

    // Check if hash is in known hashes (instant crack)
    if (knownHashes[hash.toLowerCase()]) {
      plaintext = knownHashes[hash.toLowerCase()];
      cracked = true;
      attempts = Math.floor(Math.random() * 1000000);
    } else {
      // Simulate cracking process
      const maxAttempts = attackMode === 'dictionary' ? 14344391 :
                         attackMode === 'brute' ? Math.pow(charset.length, maxLength) :
                         1000000;

      const duration = attackMode === 'dictionary' ? 3000 : 5000;
      const steps = 20;

      for (let i = 0; i < steps; i++) {
        if (!running && isBatch) break;

        await new Promise(r => setTimeout(r, duration / steps));
        attempts += Math.floor(maxAttempts / steps);
        setProgress((i + 1) / steps * 100);

        // Update hash rate
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = attempts / elapsed;
        setHashRate(Math.floor(rate));

        // Calculate ETA
        const remaining = maxAttempts - attempts;
        const etaSeconds = remaining / rate;
        const hours = Math.floor(etaSeconds / 3600);
        const minutes = Math.floor((etaSeconds % 3600) / 60);
        const seconds = Math.floor(etaSeconds % 60);
        setEta(`${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`);
      }

      // Randomly decide if crack was successful (higher chance with dictionary)
      if (attackMode === 'dictionary' && Math.random() > 0.3) {
        cracked = true;
        const commonPasswords = ['password', 'admin', 'test', 'welcome', 'qwerty', 'letmein', 'monkey', 'dragon'];
        plaintext = commonPasswords[Math.floor(Math.random() * commonPasswords.length)];
      } else if (attackMode === 'brute' && Math.random() > 0.7) {
        cracked = true;
        plaintext = Array.from({length: Math.floor(Math.random() * 4) + 4}, () =>
          charset[Math.floor(Math.random() * charset.length)]
        ).join('');
      }
    }

    const timeElapsed = (Date.now() - startTime) / 1000;

    return {
      hash,
      algorithm: selectedAlgorithm === 'auto' ? detectedAlgorithm : selectedAlgorithm,
      plaintext,
      cracked,
      attempts,
      timeElapsed
    };
  };

  const startSingleCrack = async () => {
    if (!singleHash.trim()) return;

    setRunning(true);
    setResult(null);
    setProgress(0);
    setHashRate(0);

    const result = await crackHash(singleHash.trim());
    setResult(result);
    setProgress(100);
    setRunning(false);
  };

  const startBatchCrack = async () => {
    const hashes = batchHashes.split('\n').filter(h => h.trim());
    if (hashes.length === 0) return;

    setRunning(true);
    setBatchResults([]);

    for (const hash of hashes) {
      const result = await crackHash(hash.trim(), true);
      setBatchResults(prev => [...prev, result]);
    }

    setRunning(false);
  };

  const loadDictionary = () => {
    const dict = commonDictionaries.find(d => d.name === dictionary);
    if (dict) {
      setDictionaryStats({
        loaded: 0,
        total: parseInt(dict.words),
        progress: 0
      });

      // Simulate loading
      const interval = setInterval(() => {
        setDictionaryStats(prev => {
          if (!prev) return null;
          const newLoaded = Math.min(prev.loaded + Math.floor(prev.total / 20), prev.total);
          const newProgress = (newLoaded / prev.total) * 100;

          if (newLoaded >= prev.total) {
            clearInterval(interval);
          }

          return {
            loaded: newLoaded,
            total: prev.total,
            progress: newProgress
          };
        });
      }, 100);
    }
  };

  return (
    <div className="space-y-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="bg-black/50 border border-red-900/50 grid grid-cols-4 w-full">
          <TabsTrigger value="single" className="data-[state=active]:bg-red-600">
            <Key className="w-4 h-4 mr-2" />
            Single Hash
          </TabsTrigger>
          <TabsTrigger value="batch" className="data-[state=active]:bg-red-600">
            <Database className="w-4 h-4 mr-2" />
            Batch Mode
          </TabsTrigger>
          <TabsTrigger value="config" className="data-[state=active]:bg-red-600">
            <Shield className="w-4 h-4 mr-2" />
            Configuration
          </TabsTrigger>
          <TabsTrigger value="stats" className="data-[state=active]:bg-red-600">
            <Cpu className="w-4 h-4 mr-2" />
            Statistics
          </TabsTrigger>
        </TabsList>

        {/* SINGLE HASH TAB */}
        <TabsContent value="single" className="space-y-4">
          <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
            <div className="space-y-3">
              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="text-sm text-gray-400 mb-2 block">Hash to Crack</label>
                  <Input
                    value={singleHash}
                    onChange={(e) => setSingleHash(e.target.value)}
                    placeholder="Enter hash (e.g., 5f4dcc3b5aa765d61d8327deb882cf99)"
                    className="bg-black/50 border-red-900/50 font-mono"
                  />
                  {detectedAlgorithm && (
                    <div className="mt-2">
                      <Badge variant="outline" className="border-blue-500/50 text-blue-400">
                        Detected: {detectedAlgorithm}
                      </Badge>
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Algorithm</label>
                  <Select value={selectedAlgorithm} onValueChange={setSelectedAlgorithm}>
                    <SelectTrigger className="bg-black/50 border-red-900/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto-detect</SelectItem>
                      {algorithms.map(alg => (
                        <SelectItem key={alg.id} value={alg.id}>{alg.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Attack Mode</label>
                  <Select value={attackMode} onValueChange={(val: any) => setAttackMode(val)}>
                    <SelectTrigger className="bg-black/50 border-red-900/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="dictionary">Dictionary Attack</SelectItem>
                      <SelectItem value="brute">Brute Force</SelectItem>
                      <SelectItem value="hybrid">Hybrid</SelectItem>
                      <SelectItem value="rule">Rule-based</SelectItem>
                      <SelectItem value="mask">Mask Attack</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button
                onClick={startSingleCrack}
                disabled={running || !singleHash.trim()}
                className="bg-red-600 hover:bg-red-700 w-full"
              >
                {running ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    Cracking... {progress.toFixed(1)}%
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4 mr-2" />
                    Start Cracking
                  </>
                )}
              </Button>
            </div>
          </div>

          {running && (
            <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-400">Progress</span>
                    <span className="text-sm text-red-400">{progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-black/50 rounded-full h-2 overflow-hidden">
                    <motion.div
                      className="bg-red-600 h-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <div className="text-xs text-gray-400">Hash Rate</div>
                    <div className="text-lg text-red-400">{(hashRate / 1000000).toFixed(2)}M H/s</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">ETA</div>
                    <div className="text-lg text-red-400 font-mono">{eta}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Threads</div>
                    <div className="text-lg text-red-400">{threads}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`border rounded-lg p-4 ${
                result.cracked
                  ? 'bg-green-950/20 border-green-500/30'
                  : 'bg-red-950/20 border-red-500/30'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <h4 className="text-red-50">Result</h4>
                <Badge
                  variant="outline"
                  className={
                    result.cracked
                      ? 'border-green-500/50 text-green-400'
                      : 'border-red-500/50 text-red-400'
                  }
                >
                  {result.cracked ? 'CRACKED' : 'NOT CRACKED'}
                </Badge>
              </div>

              <div className="space-y-2 text-sm">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-gray-400">Hash:</span>
                    <div className="text-red-50 font-mono text-xs break-all">{result.hash}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Algorithm:</span>
                    <div className="text-red-50">{result.algorithm}</div>
                  </div>
                </div>

                {result.cracked && result.plaintext && (
                  <div className="bg-green-950/20 rounded p-3 mt-3">
                    <span className="text-gray-400">Plaintext:</span>
                    <div className="text-green-400 font-mono text-lg mt-1">{result.plaintext}</div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2 pt-3 border-t border-red-900/30">
                  <div>
                    <span className="text-gray-400">Attempts:</span>
                    <div className="text-red-50">{result.attempts.toLocaleString()}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Time:</span>
                    <div className="text-red-50">{result.timeElapsed.toFixed(2)}s</div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
            <h4 className="text-red-50 mb-3 text-sm">Quick Test Hashes</h4>
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSingleHash('5f4dcc3b5aa765d61d8327deb882cf99')}
                className="border-red-500/50 justify-start font-mono text-xs"
              >
                MD5: 5f4dcc3b...
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSingleHash('5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8')}
                className="border-red-500/50 justify-start font-mono text-xs"
              >
                SHA1: 5baa61e4...
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSingleHash('5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8')}
                className="border-red-500/50 justify-start font-mono text-xs"
              >
                SHA256: 5e884898...
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSingleHash('8846f7eaee8fb117ad06bdd830b7586c')}
                className="border-red-500/50 justify-start font-mono text-xs"
              >
                NTLM: 8846f7ea...
              </Button>
            </div>
          </div>
        </TabsContent>

        {/* BATCH MODE TAB */}
        <TabsContent value="batch" className="space-y-4">
          <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
            <label className="text-sm text-gray-400 mb-2 block">Hash List (one per line)</label>
            <Textarea
              value={batchHashes}
              onChange={(e) => setBatchHashes(e.target.value)}
              className="bg-black/50 border-red-900/50 font-mono text-sm h-[200px]"
              placeholder="5f4dcc3b5aa765d61d8327deb882cf99&#10;098f6bcd4621d373cade4e832627b4f6&#10;827ccb0eea8a706c4c34a16891f84e7b"
            />
            <div className="flex items-center justify-between mt-3">
              <span className="text-sm text-gray-400">
                {batchHashes.split('\n').filter(h => h.trim()).length} hashes
              </span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="border-red-500/50">
                  <Upload className="w-4 h-4 mr-2" />
                  Import File
                </Button>
                <Button
                  onClick={startBatchCrack}
                  disabled={running || batchHashes.trim().length === 0}
                  className="bg-red-600 hover:bg-red-700"
                >
                  {running ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                      Cracking...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4 mr-2" />
                      Start Batch
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>

          {batchResults.length > 0 && (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-green-950/20 border border-green-500/30 rounded-lg p-3 text-center">
                  <div className="text-xs text-gray-400">Cracked</div>
                  <div className="text-2xl text-green-400">{batchResults.filter(r => r.cracked).length}</div>
                </div>
                <div className="bg-red-950/20 border border-red-500/30 rounded-lg p-3 text-center">
                  <div className="text-xs text-gray-400">Failed</div>
                  <div className="text-2xl text-red-400">{batchResults.filter(r => !r.cracked).length}</div>
                </div>
                <div className="bg-blue-950/20 border border-blue-500/30 rounded-lg p-3 text-center">
                  <div className="text-xs text-gray-400">Total</div>
                  <div className="text-2xl text-blue-400">{batchResults.length}</div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <h4 className="text-red-50">Results</h4>
                <Button variant="outline" size="sm" className="border-red-500/50">
                  <Download className="w-4 h-4 mr-2" />
                  Export Results
                </Button>
              </div>

              <ScrollArea className="h-[350px] bg-black/50 border border-red-900/30 rounded-lg">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-red-950/30 border-b border-red-900/50">
                    <tr className="text-left">
                      <th className="p-3 text-gray-400">#</th>
                      <th className="p-3 text-gray-400">Hash</th>
                      <th className="p-3 text-gray-400">Algorithm</th>
                      <th className="p-3 text-gray-400">Status</th>
                      <th className="p-3 text-gray-400">Plaintext</th>
                      <th className="p-3 text-gray-400">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.map((result, idx) => (
                      <tr key={idx} className="border-b border-red-900/20 hover:bg-red-950/20">
                        <td className="p-3 text-gray-500">{idx + 1}</td>
                        <td className="p-3 text-gray-300 font-mono text-xs truncate max-w-xs">{result.hash}</td>
                        <td className="p-3 text-gray-400">{result.algorithm}</td>
                        <td className="p-3">
                          <Badge
                            variant="outline"
                            className={
                              result.cracked
                                ? 'border-green-500/50 text-green-400'
                                : 'border-red-500/50 text-red-400'
                            }
                          >
                            {result.cracked ? 'Cracked' : 'Failed'}
                          </Badge>
                        </td>
                        <td className="p-3 text-green-400 font-mono">{result.plaintext || '-'}</td>
                        <td className="p-3 text-gray-400">{result.timeElapsed.toFixed(2)}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </ScrollArea>
            </div>
          )}
        </TabsContent>

        {/* CONFIGURATION TAB */}
        <TabsContent value="config" className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
              <h4 className="text-red-50 mb-3 text-sm">Dictionary Attack</h4>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Dictionary File</label>
                  <Select value={dictionary} onValueChange={setDictionary}>
                    <SelectTrigger className="bg-black/50 border-red-900/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {commonDictionaries.map(dict => (
                        <SelectItem key={dict.name} value={dict.name}>
                          {dict.name} ({dict.size})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  onClick={loadDictionary}
                  variant="outline"
                  className="border-red-500/50 w-full"
                >
                  <Database className="w-4 h-4 mr-2" />
                  Load Dictionary
                </Button>

                {dictionaryStats && (
                  <div className="bg-black/30 rounded p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Loading...</span>
                      <span className="text-xs text-red-400">{dictionaryStats.progress.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-black/50 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-red-600 h-full transition-all duration-300"
                        style={{ width: `${dictionaryStats.progress}%` }}
                      />
                    </div>
                    <div className="text-xs text-gray-500 mt-2">
                      {dictionaryStats.loaded.toLocaleString()} / {dictionaryStats.total.toLocaleString()} words
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
              <h4 className="text-red-50 mb-3 text-sm">Brute Force Attack</h4>
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Character Set</label>
                  <Textarea
                    value={charset}
                    onChange={(e) => setCharset(e.target.value)}
                    className="bg-black/50 border-red-900/50 font-mono text-sm h-[60px]"
                  />
                  <div className="flex gap-2 mt-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCharset('abcdefghijklmnopqrstuvwxyz')}
                      className="border-red-500/50 text-xs"
                    >
                      Lowercase
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCharset('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
                      className="border-red-500/50 text-xs"
                    >
                      Uppercase
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCharset('0123456789')}
                      className="border-red-500/50 text-xs"
                    >
                      Numbers
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCharset('!@#$%^&*()')}
                      className="border-red-500/50 text-xs"
                    >
                      Special
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Min Length</label>
                    <Input
                      type="number"
                      value={minLength}
                      onChange={(e) => setMinLength(Number(e.target.value))}
                      className="bg-black/50 border-red-900/50"
                      min="1"
                      max={maxLength}
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Max Length</label>
                    <Input
                      type="number"
                      value={maxLength}
                      onChange={(e) => setMaxLength(Number(e.target.value))}
                      className="bg-black/50 border-red-900/50"
                      min={minLength}
                      max="14"
                    />
                  </div>
                </div>

                <div className="bg-yellow-950/20 rounded p-3 text-xs text-yellow-400">
                  Keyspace: ~{Math.pow(charset.length, maxLength).toExponential(2)} combinations
                </div>
              </div>
            </div>
          </div>

          <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
            <h4 className="text-red-50 mb-3 text-sm">Performance</h4>
            <div>
              <label className="text-sm text-gray-400 mb-2 block">CPU Threads: {threads}</label>
              <input
                type="range"
                min="1"
                max="16"
                value={threads}
                onChange={(e) => setThreads(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-2">
                More threads = faster cracking but higher CPU usage
              </div>
            </div>
          </div>
        </TabsContent>

        {/* STATISTICS TAB */}
        <TabsContent value="stats" className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-2">Total Cracked</div>
              <div className="text-3xl text-green-400">{batchResults.filter(r => r.cracked).length + (result?.cracked ? 1 : 0)}</div>
            </div>
            <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-2">Success Rate</div>
              <div className="text-3xl text-blue-400">
                {batchResults.length > 0 ? ((batchResults.filter(r => r.cracked).length / batchResults.length) * 100).toFixed(1) : '0'}%
              </div>
            </div>
            <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-2">Avg. Hash Rate</div>
              <div className="text-3xl text-red-400">{(hashRate / 1000000).toFixed(2)}M H/s</div>
            </div>
          </div>

          <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
            <h4 className="text-red-50 mb-3">Algorithm Support</h4>
            <div className="grid grid-cols-4 gap-2">
              {algorithms.map(alg => (
                <div key={alg.id} className="bg-black/30 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">{alg.name}</div>
                  <Badge variant="outline" className="border-green-500/50 text-green-400 mt-2">
                    Supported
                  </Badge>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-black/50 border border-red-900/30 rounded-lg p-4">
            <h4 className="text-red-50 mb-3">Common Weak Passwords</h4>
            <div className="grid grid-cols-5 gap-2">
              {Object.values(knownHashes).slice(0, 10).map((pwd, idx) => (
                <div key={idx} className="bg-red-950/20 rounded p-2 text-center">
                  <code className="text-xs text-red-400">{pwd}</code>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-3">
              These common passwords are instantly cracked from our database
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
