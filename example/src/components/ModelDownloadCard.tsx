/* eslint-disable react/require-default-props */
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { ModelDownloader } from '../services/ModelDownloader';
import type { DownloadProgress } from '../services/ModelDownloader';

interface ModelDownloadCardProps {
  title: string;
  description: string;
  repo: string;
  filename: string;
  size: string;
  onDownloaded?: (path: string) => void;
  onInitialize?: (path: string) => void;
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginVertical: 8,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
    flex: 1,
  },
  size: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  description: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
    lineHeight: 20,
  },
  progressContainer: {
    marginBottom: 16,
  },
  progressBar: {
    height: 4,
    backgroundColor: '#E0E0E0',
    borderRadius: 2,
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 2,
  },
  progressText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  downloadButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    flex: 1,
  },
  downloadButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  downloadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
  },
  downloadingText: {
    marginLeft: 8,
    fontSize: 16,
    color: '#007AFF',
  },
  downloadedContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    flex: 1,
  },
  downloadedIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  checkmark: {
    fontSize: 20,
    color: '#4CAF50',
    marginRight: 8,
  },
  downloadedText: {
    fontSize: 16,
    color: '#4CAF50',
    fontWeight: '500',
  },
  deleteButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
  },
  deleteButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  initializeButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
    marginLeft: 8,
  },
  initializeButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  actionButtonsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
});

function ModelDownloadCard({
  title,
  description,
  repo,
  filename,
  size,
  onDownloaded: _onDownloaded,
  onInitialize,
}: ModelDownloadCardProps) {
  const [isDownloaded, setIsDownloaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [modelPath, setModelPath] = useState<string | null>(null);

  const downloader = new ModelDownloader();

    const checkIfDownloaded = React.useCallback(async () => {
    try {
      const downloaded = await ModelDownloader.isModelDownloaded(filename);
      setIsDownloaded(downloaded);

      if (downloaded) {
        const path = await ModelDownloader.getModelPath(filename);
        setModelPath(path);
      }
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  }, [filename]);

  useEffect(() => {
    checkIfDownloaded();
  }, [checkIfDownloaded]);

  const handleDownload = async () => {
    if (isDownloading) return;

    try {
      setIsDownloading(true);
      setProgress({ written: 0, total: 0, percentage: 0 });

      const path = await downloader.downloadModel(repo, filename, (prog) => {
        setProgress(prog);
      });

      setModelPath(path);
      setIsDownloaded(true);
      setProgress(null);

      Alert.alert('Success', `${title} downloaded successfully!`);
    } catch (error: any) {
      Alert.alert('Download Failed', error.message || 'Failed to download model');
      setProgress(null);
    } finally {
      setIsDownloading(false);
    }
  };

  const handleInitialize = async () => {
    if (!isDownloaded || !modelPath) {
      Alert.alert('Error', 'Model not downloaded yet.');
      return;
    }

    if (onInitialize) {
      onInitialize(modelPath);
    } else {
      Alert.alert('Error', 'No initialization handler provided.');
    }
  };

  const handleDelete = async () => {
    Alert.alert(
      'Delete Model',
      `Are you sure you want to delete ${title}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await ModelDownloader.deleteModel(filename);
              setIsDownloaded(false);
              setModelPath(null);
            } catch (error: any) {
              Alert.alert('Error', 'Failed to delete model');
            }
          },
        },
      ]
    );
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round(bytes / Math.pow(k, i) * 100) / 100} ${sizes[i]}`;
  };

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.size}>{size}</Text>
      </View>

      <Text style={styles.description}>{description}</Text>

      {progress && (
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View
              style={[styles.progressFill, { width: `${progress.percentage}%` }]}
            />
          </View>
          <Text style={styles.progressText}>
            {`${progress.percentage}% (`}
            {formatSize(progress.written)}
            {' / '}
            {formatSize(progress.total)}
            )
          </Text>
        </View>
      )}

      <View style={styles.buttonContainer}>
        {!isDownloaded && !isDownloading && (
          <TouchableOpacity style={styles.downloadButton} onPress={handleDownload}>
            <Text style={styles.downloadButtonText}>Download</Text>
          </TouchableOpacity>
        )}

        {isDownloading && (
          <View style={styles.downloadingContainer}>
            <ActivityIndicator size="small" color="#007AFF" />
            <Text style={styles.downloadingText}>Downloading...</Text>
          </View>
        )}

        {isDownloaded && !isDownloading && (
          <View style={styles.downloadedContainer}>
            <View style={styles.downloadedIndicator}>
              <Text style={styles.checkmark}>✓</Text>
              <Text style={styles.downloadedText}>Downloaded</Text>
            </View>
            <View style={styles.actionButtonsContainer}>
              <TouchableOpacity style={styles.deleteButton} onPress={handleDelete}>
                <Text style={styles.deleteButtonText}>Delete</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.initializeButton} onPress={handleInitialize}>
                <Text style={styles.initializeButtonText}>Initialize</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    </View>
  );
}

export default ModelDownloadCard;

// TTS-specific download card that handles both TTS model and vocoder together
interface TTSModelDownloadCardProps {
  title: string;
  description: string;
  repo: string;
  filename: string;
  size: string;
  vocoder: {
    repo: string;
    filename: string;
    size: string;
  };
  onInitialize: (ttsPath: string, vocoderPath: string) => void;
}

export function TTSModelDownloadCard({
  title,
  description,
  repo,
  filename,
  size,
  vocoder,
  onInitialize,
}: TTSModelDownloadCardProps) {
  const [isDownloaded, setIsDownloaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [ttsPath, setTtsPath] = useState<string | null>(null);
  const [vocoderPath, setVocoderPath] = useState<string | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<string>('');

  const downloader = new ModelDownloader();

  const checkIfDownloaded = React.useCallback(async () => {
    try {
      const ttsDownloaded = await ModelDownloader.isModelDownloaded(filename);
      const vocoderDownloaded = await ModelDownloader.isModelDownloaded(vocoder.filename);

      const bothDownloaded = ttsDownloaded && vocoderDownloaded;
      setIsDownloaded(bothDownloaded);

      if (ttsDownloaded) {
        const path = await ModelDownloader.getModelPath(filename);
        setTtsPath(path);
      }

      if (vocoderDownloaded) {
        const path = await ModelDownloader.getModelPath(vocoder.filename);
        setVocoderPath(path);
      }
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  }, [filename, vocoder.filename]);

  useEffect(() => {
    checkIfDownloaded();
  }, [checkIfDownloaded]);

  const handleDownload = async () => {
    if (isDownloading) return;

    try {
      setIsDownloading(true);
      setProgress({ written: 0, total: 0, percentage: 0 });

      // Download TTS model first
      setDownloadStatus('Downloading TTS model...');
      const ttsModelPath = await downloader.downloadModel(repo, filename, (prog) => {
        setProgress({
          ...prog,
          percentage: Math.round(prog.percentage * 0.7), // 70% for TTS model
        });
      });
      setTtsPath(ttsModelPath);

      // Download vocoder model
      setDownloadStatus('Downloading vocoder...');
      const vocoderModelPath = await downloader.downloadModel(vocoder.repo, vocoder.filename, (prog) => {
        setProgress({
          ...prog,
          percentage: 70 + Math.round(prog.percentage * 0.3), // 30% for vocoder
        });
      });
      setVocoderPath(vocoderModelPath);

      setIsDownloaded(true);
      setProgress(null);
      setDownloadStatus('');

      Alert.alert('Success', `${title} downloaded successfully!`);
    } catch (error: any) {
      Alert.alert('Download Failed', error.message || 'Failed to download models');
      setProgress(null);
      setDownloadStatus('');
    } finally {
      setIsDownloading(false);
    }
  };

  const handleInitialize = async () => {
    if (!isDownloaded || !ttsPath || !vocoderPath) {
      Alert.alert('Error', 'Models not downloaded yet.');
      return;
    }

    if (onInitialize) {
      onInitialize(ttsPath, vocoderPath);
    } else {
      Alert.alert('Error', 'No initialization handler provided.');
    }
  };

  const handleDelete = async () => {
    Alert.alert(
      'Delete Models',
      `Are you sure you want to delete ${title}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await ModelDownloader.deleteModel(filename);
              await ModelDownloader.deleteModel(vocoder.filename);
              setIsDownloaded(false);
              setTtsPath(null);
              setVocoderPath(null);
            } catch (error: any) {
              Alert.alert('Error', 'Failed to delete models');
            }
          },
        },
      ]
    );
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round(bytes / Math.pow(k, i) * 100) / 100} ${sizes[i]}`;
  };

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.size}>{size}</Text>
      </View>

      <Text style={styles.description}>{description}</Text>

      {progress && (
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View
              style={[styles.progressFill, { width: `${progress.percentage}%` }]}
            />
          </View>
          <Text style={styles.progressText}>
            {downloadStatus}
            {' '}
            {`${progress.percentage}%`}
          </Text>
          {progress.total > 0 && (
            <Text style={styles.progressText}>
              (
              {formatSize(progress.written)}
              {' / '}
              {formatSize(progress.total)}
              )
            </Text>
          )}
        </View>
      )}

      <View style={styles.buttonContainer}>
        {!isDownloaded && !isDownloading && (
          <TouchableOpacity style={styles.downloadButton} onPress={handleDownload}>
            <Text style={styles.downloadButtonText}>Download Both Models</Text>
          </TouchableOpacity>
        )}

        {isDownloading && (
          <View style={styles.downloadingContainer}>
            <ActivityIndicator size="small" color="#007AFF" />
            <Text style={styles.downloadingText}>Downloading...</Text>
          </View>
        )}

        {isDownloaded && !isDownloading && (
          <View style={styles.downloadedContainer}>
            <View style={styles.downloadedIndicator}>
              <Text style={styles.checkmark}>✓</Text>
              <Text style={styles.downloadedText}>Downloaded</Text>
            </View>
            <View style={styles.actionButtonsContainer}>
              <TouchableOpacity style={styles.deleteButton} onPress={handleDelete}>
                <Text style={styles.deleteButtonText}>Delete</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.initializeButton} onPress={handleInitialize}>
                <Text style={styles.initializeButtonText}>Initialize</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    </View>
  );
}

// VLM-specific download card that handles both model and mmproj files
interface VLMModelDownloadCardProps {
  title: string;
  description: string;
  repo: string;
  filename: string;
  mmproj: string;
  size: string;
  onInitialize: (modelPath: string, mmprojPath: string) => void;
}

export function VLMModelDownloadCard({
  title,
  description,
  repo,
  filename,
  mmproj,
  size,
  onInitialize,
}: VLMModelDownloadCardProps) {
  const [isDownloaded, setIsDownloaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [modelPath, setModelPath] = useState<string | null>(null);
  const [mmprojPath, setMmprojPath] = useState<string | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<string>('');

  const downloader = new ModelDownloader();

  const checkIfDownloaded = React.useCallback(async () => {
    try {
      const modelDownloaded = await ModelDownloader.isModelDownloaded(filename);
      const mmprojDownloaded = await ModelDownloader.isModelDownloaded(mmproj);

      const bothDownloaded = modelDownloaded && mmprojDownloaded;
      setIsDownloaded(bothDownloaded);

      if (modelDownloaded) {
        const path = await ModelDownloader.getModelPath(filename);
        setModelPath(path);
      }

      if (mmprojDownloaded) {
        const path = await ModelDownloader.getModelPath(mmproj);
        setMmprojPath(path);
      }
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  }, [filename, mmproj]);

  useEffect(() => {
    checkIfDownloaded();
  }, [checkIfDownloaded]);

  const handleDownload = async () => {
    if (isDownloading) return;

    try {
      setIsDownloading(true);
      setProgress({ written: 0, total: 0, percentage: 0 });

      // Download main model first
      setDownloadStatus('Downloading VLM model...');
      const vlmModelPath = await downloader.downloadModel(repo, filename, (prog) => {
        setProgress({
          ...prog,
          percentage: Math.round(prog.percentage * 0.8), // 80% for main model
        });
      });
      setModelPath(vlmModelPath);

      // Download mmproj
      setDownloadStatus('Downloading mmproj...');
      const vlmMmprojPath = await downloader.downloadModel(repo, mmproj, (prog) => {
        setProgress({
          ...prog,
          percentage: 80 + Math.round(prog.percentage * 0.2), // 20% for mmproj
        });
      });
      setMmprojPath(vlmMmprojPath);

      setIsDownloaded(true);
      setProgress(null);
      setDownloadStatus('');

      Alert.alert('Success', `${title} downloaded successfully!`);
    } catch (error: any) {
      Alert.alert('Download Failed', error.message || 'Failed to download models');
      setProgress(null);
      setDownloadStatus('');
    } finally {
      setIsDownloading(false);
    }
  };

  const handleInitialize = async () => {
    if (!isDownloaded || !modelPath || !mmprojPath) {
      Alert.alert('Error', 'Models not downloaded yet.');
      return;
    }

    if (onInitialize) {
      onInitialize(modelPath, mmprojPath);
    } else {
      Alert.alert('Error', 'No initialization handler provided.');
    }
  };

  const handleDelete = async () => {
    Alert.alert(
      'Delete Models',
      `Are you sure you want to delete ${title}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await ModelDownloader.deleteModel(filename);
              await ModelDownloader.deleteModel(mmproj);
              setIsDownloaded(false);
              setModelPath(null);
              setMmprojPath(null);
            } catch (error: any) {
              Alert.alert('Error', 'Failed to delete models');
            }
          },
        },
      ]
    );
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round(bytes / Math.pow(k, i) * 100) / 100} ${sizes[i]}`;
  };

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.size}>{size}</Text>
      </View>

      <Text style={styles.description}>{description}</Text>

      {progress && (
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View
              style={[styles.progressFill, { width: `${progress.percentage}%` }]}
            />
          </View>
          <Text style={styles.progressText}>
            {downloadStatus}
            {' '}
            {`${progress.percentage}%`}
          </Text>
          {progress.total > 0 && (
            <Text style={styles.progressText}>
              (
              {formatSize(progress.written)}
              {' / '}
              {formatSize(progress.total)}
              )
            </Text>
          )}
        </View>
      )}

      <View style={styles.buttonContainer}>
        {!isDownloaded && !isDownloading && (
          <TouchableOpacity style={styles.downloadButton} onPress={handleDownload}>
            <Text style={styles.downloadButtonText}>Download VLM & MMProj</Text>
          </TouchableOpacity>
        )}

        {isDownloading && (
          <View style={styles.downloadingContainer}>
            <ActivityIndicator size="small" color="#007AFF" />
            <Text style={styles.downloadingText}>Downloading...</Text>
          </View>
        )}

        {isDownloaded && !isDownloading && (
          <View style={styles.downloadedContainer}>
            <View style={styles.downloadedIndicator}>
              <Text style={styles.checkmark}>✓</Text>
              <Text style={styles.downloadedText}>Downloaded</Text>
            </View>
            <View style={styles.actionButtonsContainer}>
              <TouchableOpacity style={styles.deleteButton} onPress={handleDelete}>
                <Text style={styles.deleteButtonText}>Delete</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.initializeButton} onPress={handleInitialize}>
                <Text style={styles.initializeButtonText}>Initialize</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    </View>
  );
}
