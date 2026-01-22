#!/usr/bin/env node
/**
 * Epistemic DJ - YouTube Audio Analyzer
 * Downloads audio from YouTube and extracts epistemic vectors
 *
 * Usage: node youtube-to-vectors.js <youtube-url>
 */

import { spawn, execSync } from 'child_process';
import { readFileSync, unlinkSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TEMP_DIR = join(__dirname, '../../.temp');

// Ensure temp directory exists
if (!existsSync(TEMP_DIR)) {
  mkdirSync(TEMP_DIR, { recursive: true });
}

/**
 * Download audio from YouTube
 */
async function downloadAudio(url) {
  const outputPath = join(TEMP_DIR, 'audio.wav');

  console.error('📥 Downloading audio from YouTube...');

  try {
    // Download and convert to WAV in one step
    execSync(`yt-dlp -x --audio-format wav -o "${TEMP_DIR}/audio.%(ext)s" "${url}" 2>&1`, {
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 120000, // 2 minute timeout
    });

    // yt-dlp might output to different filename
    const files = execSync(`ls ${TEMP_DIR}/audio.* 2>/dev/null || true`).toString().trim().split('\n');
    const audioFile = files.find(f => f.endsWith('.wav') || f.endsWith('.opus') || f.endsWith('.m4a') || f.endsWith('.webm'));

    if (!audioFile || !existsSync(audioFile)) {
      throw new Error('Failed to download audio');
    }

    // Convert to WAV if not already
    if (!audioFile.endsWith('.wav')) {
      console.error('🔄 Converting to WAV...');
      execSync(`ffmpeg -y -i "${audioFile}" -ar 44100 -ac 1 "${outputPath}" 2>/dev/null`);
      unlinkSync(audioFile);
    }

    return outputPath;
  } catch (error) {
    console.error('Error downloading:', error.message);
    throw error;
  }
}

/**
 * Analyze audio file using ffmpeg's audio filters for feature extraction
 * This is a simplified approach - for full Meyda analysis, use the browser version
 */
async function analyzeAudio(audioPath) {
  console.error('🔬 Analyzing audio features...');

  const features = {};

  try {
    // Get audio stats using ffmpeg
    const statsOutput = execSync(
      `ffmpeg -i "${audioPath}" -af "volumedetect" -f null - 2>&1 | grep -E "mean_volume|max_volume"`,
      { encoding: 'utf-8' }
    );

    // Parse volume stats
    const meanMatch = statsOutput.match(/mean_volume:\s*([-\d.]+)/);
    const maxMatch = statsOutput.match(/max_volume:\s*([-\d.]+)/);

    features.meanVolume = meanMatch ? parseFloat(meanMatch[1]) : -20;
    features.maxVolume = maxMatch ? parseFloat(maxMatch[1]) : -3;
    features.dynamicRange = features.maxVolume - features.meanVolume;

    // Get tempo/BPM estimate using ffmpeg's ebur128 filter
    const loudnessOutput = execSync(
      `ffmpeg -i "${audioPath}" -af "ebur128=peak=true" -f null - 2>&1 | tail -20`,
      { encoding: 'utf-8', timeout: 60000 }
    );

    // Parse integrated loudness
    const integratedMatch = loudnessOutput.match(/I:\s*([-\d.]+)/);
    features.integratedLoudness = integratedMatch ? parseFloat(integratedMatch[1]) : -14;

    // Get spectral info using showspectrumpic (simplified)
    // For actual spectral centroid, we'd need proper DSP

    // Get duration
    const duration = parseFloat(execSync(
      `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${audioPath}"`,
      { encoding: 'utf-8' }
    ).trim());
    features.duration = duration;

    // Use aubio for proper BPM detection
    try {
      const aubioOutput = execSync(
        `aubiotrack -i "${audioPath}" 2>/dev/null | wc -l`,
        { encoding: 'utf-8', timeout: 120000 }
      );
      const beatCount = parseInt(aubioOutput.trim()) || 0;

      if (beatCount > 10 && duration > 30) {
        // Calculate BPM from beat count and duration
        features.estimatedBPM = Math.round((beatCount / duration) * 60);
        features.beatCount = beatCount;
      } else {
        // Fallback: try aubioonset for onset-based estimation
        const onsetOutput = execSync(
          `aubioonset -i "${audioPath}" 2>/dev/null | wc -l`,
          { encoding: 'utf-8', timeout: 120000 }
        );
        const onsetCount = parseInt(onsetOutput.trim()) || 0;
        features.estimatedBPM = onsetCount > 0 ? Math.round((onsetCount / duration) * 60 / 4) : 120;
        features.onsetCount = onsetCount;
      }
    } catch (aubioError) {
      console.error('Warning: aubio analysis failed, using fallback');
      features.estimatedBPM = 120; // Default fallback
    }

  } catch (error) {
    console.error('Warning: Some analysis failed:', error.message);
    // Use defaults
    features.meanVolume = features.meanVolume || -20;
    features.maxVolume = features.maxVolume || -3;
    features.dynamicRange = features.dynamicRange || 17;
    features.integratedLoudness = features.integratedLoudness || -14;
    features.estimatedBPM = features.estimatedBPM || 120;
    features.duration = features.duration || 180;
  }

  return features;
}

/**
 * Map audio features to epistemic vectors
 */
function mapToEpistemicVectors(features) {
  const {
    meanVolume = -20,
    maxVolume = -3,
    dynamicRange = 17,
    integratedLoudness = -14,
    estimatedBPM = 120,
    duration = 180,
  } = features;

  // Normalize features to 0-1 scale

  // Engagement: Based on tempo (60-180 BPM → 0-1)
  const engagement = Math.max(0, Math.min(1, (estimatedBPM - 60) / 120));

  // Impact: Based on loudness (-40 to 0 dB → 0-1)
  const impact = Math.max(0, Math.min(1, (integratedLoudness + 40) / 40));

  // Change: Based on dynamic range (0-30 dB → 0-1)
  const change = Math.max(0, Math.min(1, dynamicRange / 30));

  // Clarity: Inverse of dynamic range (more consistent = clearer)
  const clarity = 1 - (change * 0.5);

  // Signal density: Higher BPM + louder = denser
  const signal = (engagement + impact) / 2;

  // Know: Based on duration (longer tracks = more developed ideas)
  const know = Math.max(0.3, Math.min(0.9, duration / 300));

  // Uncertainty: Inverse of clarity
  const uncertainty = 1 - clarity;

  // Coherence: Based on how "together" the track sounds
  // (simplified - would need harmonic analysis for real)
  const coherence = Math.max(0.4, Math.min(0.9, 1 - (dynamicRange / 40)));

  // State: Based on overall energy
  const state = (engagement + impact + signal) / 3;

  // Completion: Higher tempo + louder often indicates climactic sections
  const completion = (engagement * 0.6 + impact * 0.4);

  // Context: Longer duration = more context established
  const context = know;

  // Density: Based on loudness variance
  const density = signal;

  // Do: Similar to engagement (action-oriented)
  const doVector = engagement;

  return {
    know: parseFloat(know.toFixed(3)),
    uncertainty: parseFloat(uncertainty.toFixed(3)),
    engagement: parseFloat(engagement.toFixed(3)),
    clarity: parseFloat(clarity.toFixed(3)),
    coherence: parseFloat(coherence.toFixed(3)),
    signal: parseFloat(signal.toFixed(3)),
    density: parseFloat(density.toFixed(3)),
    state: parseFloat(state.toFixed(3)),
    change: parseFloat(change.toFixed(3)),
    completion: parseFloat(completion.toFixed(3)),
    impact: parseFloat(impact.toFixed(3)),
    context: parseFloat(context.toFixed(3)),
    do: parseFloat(doVector.toFixed(3)),
  };
}

/**
 * Generate interpretation text
 */
function interpretVectors(vectors, features) {
  const interpretations = [];

  if (vectors.engagement > 0.7) {
    interpretations.push(`High energy track (~${features.estimatedBPM} BPM) - driving, active`);
  } else if (vectors.engagement < 0.3) {
    interpretations.push(`Slow, contemplative track (~${features.estimatedBPM} BPM)`);
  } else {
    interpretations.push(`Medium tempo (~${features.estimatedBPM} BPM)`);
  }

  if (vectors.impact > 0.7) {
    interpretations.push('Loud, powerful presence');
  } else if (vectors.impact < 0.3) {
    interpretations.push('Subtle, quiet dynamics');
  }

  if (vectors.change > 0.6) {
    interpretations.push('High dynamic variation - dramatic shifts');
  } else if (vectors.change < 0.3) {
    interpretations.push('Consistent dynamics - steady state');
  }

  if (vectors.coherence > 0.7) {
    interpretations.push('Cohesive, unified sound');
  }

  return interpretations;
}

/**
 * Main
 */
async function main() {
  const url = process.argv[2];

  if (!url) {
    console.log(`
🎧 Epistemic DJ - YouTube Audio Analyzer

Usage: node youtube-to-vectors.js <youtube-url>

Example:
  node youtube-to-vectors.js "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

This will:
  1. Download the audio from YouTube
  2. Analyze audio features (tempo, loudness, dynamics)
  3. Map to epistemic vectors
  4. Output JSON you can use with generate_pattern

Note: For full spectral analysis (Meyda.js), use the browser analyzer.
`);
    process.exit(1);
  }

  try {
    // Download
    const audioPath = await downloadAudio(url);

    // Analyze
    const features = await analyzeAudio(audioPath);
    console.error('📊 Raw features:', JSON.stringify(features, null, 2));

    // Map to vectors
    const vectors = mapToEpistemicVectors(features);

    // Interpret
    const interpretations = interpretVectors(vectors, features);

    // Output
    const result = {
      url,
      features,
      vectors,
      interpretation: interpretations,
    };

    console.log(JSON.stringify(result, null, 2));

    // Cleanup
    if (existsSync(audioPath)) {
      unlinkSync(audioPath);
    }

  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

main();
