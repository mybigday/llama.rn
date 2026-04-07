#!/usr/bin/env node

const crypto = require('crypto')
const fs = require('fs')
const http = require('http')
const https = require('https')
const os = require('os')
const path = require('path')
const { spawnSync } = require('child_process')
const { pipeline } = require('stream/promises')

const MAX_REDIRECTS = 5
const packageRoot = path.resolve(__dirname, '..')
const packageJson = JSON.parse(
  fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8'),
)
const manifest = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'native-artifacts.json'), 'utf8'),
)
const force = process.argv.includes('--force')

function normalizeRepositoryUrl(repository) {
  const rawUrl =
    typeof repository === 'string' ? repository : repository?.url || ''

  return rawUrl.replace(/^git\+/, '').replace(/\.git$/, '').replace(/\/$/, '')
}

const releaseBaseUrl = `${normalizeRepositoryUrl(packageJson.repository)}/releases/download/v${packageJson.version}`

function resolvePackagePath(relativePath) {
  return path.join(packageRoot, relativePath)
}

function targetExists(relativePath) {
  return fs.existsSync(resolvePackagePath(relativePath))
}

function readMarker(markerPath) {
  const absolutePath = resolvePackagePath(markerPath)

  if (!fs.existsSync(absolutePath)) {
    return null
  }

  return fs.readFileSync(absolutePath, 'utf8').trim()
}

function isArtifactInstalled(artifact) {
  return (
    targetExists(artifact.relativePath) &&
    readMarker(artifact.markerPath) === artifact.sha256
  )
}

function hasConfiguredManifest(artifacts) {
  return artifacts.every(
    (artifact) =>
      typeof artifact.sha256 === 'string' &&
      /^[\da-f]{64}$/i.test(artifact.sha256),
  )
}

function writeMarker(markerPath, sha256) {
  const absolutePath = resolvePackagePath(markerPath)
  fs.mkdirSync(path.dirname(absolutePath), { recursive: true })
  fs.writeFileSync(absolutePath, `${sha256}\n`)
}

function removeIfExists(relativePath) {
  fs.rmSync(resolvePackagePath(relativePath), {
    force: true,
    recursive: true,
  })
}

function getTransport(url) {
  return new URL(url).protocol === 'http:' ? http : https
}

function request(url, redirectCount = 0) {
  return new Promise((resolve, reject) => {
    const transport = getTransport(url)

    const req = transport.get(
      url,
      {
        headers: {
          'User-Agent': 'llama.rn postinstall',
        },
      },
      (response) => {
        const { statusCode = 0, headers } = response

        if (
          statusCode >= 300 &&
          statusCode < 400 &&
          headers.location
        ) {
          if (redirectCount >= MAX_REDIRECTS) {
            response.resume()
            reject(new Error(`Too many redirects while downloading ${url}`))
            return
          }

          response.resume()
          resolve(
            request(new URL(headers.location, url).toString(), redirectCount + 1),
          )
          return
        }

        if (statusCode !== 200) {
          response.resume()
          reject(
            new Error(`Download failed for ${url} with status code ${statusCode}`),
          )
          return
        }

        resolve(response)
      },
    )

    req.on('error', reject)
  })
}

async function downloadArchive(url, outputPath) {
  const response = await request(url)
  const hash = crypto.createHash('sha256')
  const output = fs.createWriteStream(outputPath)

  response.on('data', (chunk) => {
    hash.update(chunk)
  })

  await pipeline(response, output)
  return hash.digest('hex')
}

function extractArchive(archivePath) {
  const result = spawnSync('tar', ['-xzf', archivePath, '-C', packageRoot], {
    stdio: 'inherit',
  })

  if (result.error) {
    throw new Error(
      `Failed to extract ${path.basename(archivePath)}: ${result.error.message}`,
    )
  }

  if (result.status !== 0) {
    throw new Error(
      `tar exited with status ${result.status} while extracting ${archivePath}`,
    )
  }
}

async function installArtifact(artifact) {
  const url = `${releaseBaseUrl}/${artifact.assetName}`
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llama-rn-'))
  const archivePath = path.join(tempDir, artifact.assetName)

  try {
    console.log(`llama.rn: downloading ${artifact.assetName}`)
    const checksum = await downloadArchive(url, archivePath)

    if (checksum !== artifact.sha256) {
      throw new Error(
        `SHA-256 mismatch for ${artifact.assetName}: expected ${artifact.sha256}, received ${checksum}`,
      )
    }

    removeIfExists(artifact.relativePath)
    removeIfExists(artifact.markerPath)
    extractArchive(archivePath)

    if (!targetExists(artifact.relativePath)) {
      throw new Error(
        `Missing extracted native artifact at ${artifact.relativePath}`,
      )
    }

    writeMarker(artifact.markerPath, artifact.sha256)
    console.log(`llama.rn: installed ${artifact.relativePath}`)
  } finally {
    fs.rmSync(tempDir, { force: true, recursive: true })
  }
}

async function main() {
  if (process.env.RNLLAMA_SKIP_POSTINSTALL === '1') {
    console.log(
      'llama.rn: skipping native artifact download because RNLLAMA_SKIP_POSTINSTALL=1',
    )
    return
  }

  const artifacts = manifest.artifacts || []

  if (!hasConfiguredManifest(artifacts)) {
    const allTargetsPresent = artifacts.every((artifact) =>
      targetExists(artifact.relativePath),
    )

    if (!allTargetsPresent) {
      console.warn(
        'llama.rn: native artifact manifest is not configured for this checkout; skipping download.',
      )
    }

    return
  }

  const artifactsToInstall = artifacts.filter((artifact) => {
    if (!force && isArtifactInstalled(artifact)) {
      console.log(`llama.rn: using cached ${artifact.relativePath}`)
      return false
    }

    return true
  })

  await Promise.all(artifactsToInstall.map((artifact) => installArtifact(artifact)))
}

main().catch((error) => {
  console.error(`llama.rn: ${error.message}`)
  process.exitCode = 1
})
