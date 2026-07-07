import ReactNativeBlobUtil from 'react-native-blob-util'
import { Platform } from 'react-native'

// Decode a base64-encoded 16-bit little-endian PCM buffer (as bundled in
// example/src/assets/voices/*.ts) into a Float32Array normalized to [-1, 1].
// Whitespace inside the base64 string is stripped first so callers can keep
// the source asset wrapped for readability.
export const decodeBase64Pcm16 = (b64: string): Float32Array => {
  const clean = b64.replace(/\s+/g, '')
  const binStr = global.atob
    ? global.atob(clean)
    : Buffer.from(clean, 'base64').toString('binary')
  const bytes = new Uint8Array(binStr.length)
  for (let i = 0; i < binStr.length; i += 1) bytes[i] = binStr.charCodeAt(i)
  const view = new DataView(bytes.buffer)
  const nSamples = bytes.length >> 1
  const out = new Float32Array(nSamples)
  for (let i = 0; i < nSamples; i += 1) {
    out[i] = view.getInt16(i * 2, true) / 0x8000
  }
  return out
}

// WAV file creation utility
export const createWavFile = (audioData: Float32Array, sampleRate: number, bitDepth: number = 16): ArrayBuffer => {
  const numChannels = 1 // Mono
  const bytesPerSample = bitDepth / 8
  const blockAlign = numChannels * bytesPerSample
  const byteRate = sampleRate * blockAlign
  const dataSize = audioData.length * bytesPerSample
  const fileSize = 44 + dataSize // WAV header is 44 bytes

  const arrayBuffer = new ArrayBuffer(fileSize)
  const view = new DataView(arrayBuffer)

  // WAV Header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i += 1) {
      view.setUint8(offset + i, string.charCodeAt(i))
    }
  }

  writeString(0, 'RIFF') // ChunkID
  view.setUint32(4, fileSize - 8, true) // ChunkSize
  writeString(8, 'WAVE') // Format
  writeString(12, 'fmt ') // Subchunk1ID
  view.setUint32(16, 16, true) // Subchunk1Size
  view.setUint16(20, 1, true) // AudioFormat (PCM)
  view.setUint16(22, numChannels, true) // NumChannels
  view.setUint32(24, sampleRate, true) // SampleRate
  view.setUint32(28, byteRate, true) // ByteRate
  view.setUint16(32, blockAlign, true) // BlockAlign
  view.setUint16(34, bitDepth, true) // BitsPerSample
  writeString(36, 'data') // Subchunk2ID
  view.setUint32(40, dataSize, true) // Subchunk2Size

  // Convert Float32Array to 16-bit PCM
  let offset = 44
  for (let i = 0; i < audioData.length; i += 1) {
    // Clamp and convert to 16-bit signed integer
    const sample = Math.max(-1, Math.min(1, audioData[i] || 0))
    const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF
    view.setInt16(offset, intSample, true)
    offset += 2
  }

  return arrayBuffer
}

// Auto-dump the freshly generated PCM to a stable path so the host adb
// verification script can pull it off the device without user interaction:
//   Android: /sdcard/Download/rnllama-tts-verify.wav (accessible via `adb pull`)
//   iOS:     ${DocumentDir}/rnllama-tts-verify.wav
// Returns the final on-disk path (or null on failure).  Non-fatal — logs a
// warning if it can't write instead of surfacing to the UI.
export const dumpTtsWavToDisk = async (
  audioData: Float32Array,
  sampleRate: number,
  filename: string = 'rnllama-tts-verify.wav',
): Promise<string | null> => {
  try {
    const wavBuffer = createWavFile(audioData, sampleRate, 16)
    const bytes = new Uint8Array(wavBuffer)
    let binary = ''
    for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]!)
    const base64Data = ReactNativeBlobUtil.base64.encode(binary)

    const dir =
      Platform.OS === 'android'
        ? ReactNativeBlobUtil.fs.dirs.DownloadDir
        : ReactNativeBlobUtil.fs.dirs.DocumentDir
    const path = `${dir}/${filename}`
    await ReactNativeBlobUtil.fs.writeFile(path, base64Data, 'base64')
    // eslint-disable-next-line no-console
    console.log(`[tts-verify] wrote ${path} (${bytes.length} bytes, ${sampleRate} Hz)`)
    return path
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[tts-verify] failed to auto-dump WAV:', e)
    return null
  }
}
