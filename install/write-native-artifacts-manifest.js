#!/usr/bin/env node

const crypto = require('crypto')
const fs = require('fs')
const path = require('path')

const installDir = __dirname
const packageRoot = path.resolve(installDir, '..')
const manifestPath = path.join(installDir, 'native-artifacts.json')

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))

function sha256File(filePath) {
  const hash = crypto.createHash('sha256')
  hash.update(fs.readFileSync(filePath))
  return hash.digest('hex')
}

function resolveArchivePath(assetName) {
  return path.join(packageRoot, assetName)
}

manifest.artifacts.forEach((artifact) => {
  const archivePath = resolveArchivePath(artifact.assetName)

  if (!fs.existsSync(archivePath)) {
    throw new Error(`Missing native artifact archive: ${archivePath}`)
  }

  artifact.sha256 = sha256File(archivePath)
})

fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
