/**
 * Epistemic DJ - Pattern Generator
 * Maps 13 epistemic vectors to Strudel live coding patterns
 */

// Musical scales mapped to epistemic states
const SCALES = {
  certain: ['c', 'd', 'e', 'g', 'a'],           // Pentatonic - stable, confident
  exploring: ['c', 'd', 'eb', 'f', 'g', 'ab', 'bb'], // Dorian - curious, searching
  uncertain: ['c', 'db', 'eb', 'e', 'gb', 'g', 'a', 'bb'], // Diminished - tense
  resolving: ['c', 'd', 'e', 'f', 'g', 'a', 'b'],  // Major - clarity, resolution
  deep: ['c', 'eb', 'f', 'gb', 'g', 'bb'],       // Blues - depth, complexity
};

// Drum patterns by engagement level
const DRUM_PATTERNS = {
  low: 'bd ~ ~ ~ sd ~ ~ ~',           // Sparse, contemplative
  medium: 'bd ~ sd ~ bd ~ sd ~',       // Basic groove
  high: 'bd bd sd ~ bd ~ sd bd',       // Driving
  intense: 'bd*2 [~ bd] sd [bd sd]',   // Complex, energetic
};

// Hi-hat patterns by signal density
const HIHAT_PATTERNS = {
  sparse: 'hh ~ ~ ~',
  normal: 'hh*4',
  dense: 'hh*8',
  chaotic: 'hh*16',
};

/**
 * Map epistemic vectors to musical parameters
 */
function mapVectorsToParams(vectors) {
  const {
    know = 0.5,
    uncertainty = 0.5,
    engagement = 0.5,
    clarity = 0.5,
    coherence = 0.5,
    signal = 0.5,
    density = 0.5,
    state = 0.5,
    change = 0.5,
    completion = 0.5,
    impact = 0.5,
    context = 0.5,
    do: doVector = 0.5,
  } = vectors;

  return {
    // Tempo: 60-140 BPM based on engagement
    tempo: Math.round(60 + (engagement * 80)),

    // Scale selection based on know vs uncertainty
    scale: selectScale(know, uncertainty),

    // Octave range based on context breadth
    octaveBase: Math.floor(3 + (context * 2)),
    octaveRange: Math.ceil(1 + (density * 2)),

    // Note density based on signal
    noteDensity: signal,

    // Rhythmic complexity based on coherence (inverse)
    rhythmComplexity: 1 - coherence,

    // Filter cutoff based on clarity (dark to bright)
    cutoff: Math.round(200 + (clarity * 4000)),

    // Reverb/space based on state (grounded vs expansive)
    roomSize: 0.2 + (state * 0.6),

    // Pattern variation rate based on change
    variationRate: change,

    // Build-up intensity based on completion
    intensity: completion,

    // Overall gain based on impact
    gain: 0.3 + (impact * 0.5),

    // Probability/degradation based on uncertainty
    degradeBy: uncertainty * 0.4,

    // Engagement level for drum selection
    engagementLevel: engagement < 0.3 ? 'low' :
                     engagement < 0.6 ? 'medium' :
                     engagement < 0.8 ? 'high' : 'intense',

    // Signal density for hihat
    signalLevel: signal < 0.25 ? 'sparse' :
                 signal < 0.5 ? 'normal' :
                 signal < 0.75 ? 'dense' : 'chaotic',
  };
}

/**
 * Select scale based on know/uncertainty balance
 */
function selectScale(know, uncertainty) {
  const confidence = know - uncertainty;

  if (confidence > 0.3) return SCALES.certain;
  if (confidence > 0) return SCALES.resolving;
  if (confidence > -0.2) return SCALES.exploring;
  if (confidence > -0.4) return SCALES.deep;
  return SCALES.uncertain;
}

/**
 * Generate a melodic pattern
 */
function generateMelody(params) {
  const { scale, octaveBase, noteDensity, degradeBy, variationRate } = params;

  // Select notes from scale with varying density
  const noteCount = Math.ceil(4 + (noteDensity * 12));
  const notes = [];

  for (let i = 0; i < noteCount; i++) {
    const note = scale[Math.floor(Math.random() * scale.length)];
    const octave = octaveBase + Math.floor(Math.random() * params.octaveRange);
    notes.push(`${note}${octave}`);
  }

  // Create pattern with rests based on density
  const pattern = notes.map(n => Math.random() > noteDensity * 0.5 ? n : '~').join(' ');

  // Add variation and degradation
  let melody = `note("${pattern}")`;

  if (degradeBy > 0.1) {
    melody += `.degradeBy(${degradeBy.toFixed(2)})`;
  }

  if (variationRate > 0.3) {
    melody += `.jux(rev)`;
  }

  return melody;
}

/**
 * Generate drum pattern
 */
function generateDrums(params) {
  const { engagementLevel, signalLevel, degradeBy } = params;

  const kick = DRUM_PATTERNS[engagementLevel];
  const hihat = HIHAT_PATTERNS[signalLevel];

  let drums = `stack(
    s("${kick}").gain(0.8),
    s("${hihat}").gain(0.4)
  )`;

  if (degradeBy > 0.2) {
    drums = `${drums}.degradeBy(${(degradeBy * 0.5).toFixed(2)})`;
  }

  return drums;
}

/**
 * Generate bass line
 */
function generateBass(params) {
  const { scale, octaveBase, noteDensity, tempo } = params;

  // Bass uses root notes mostly
  const rootNote = scale[0];
  const fifthNote = scale[Math.min(4, scale.length - 1)];

  const pattern = tempo > 100
    ? `${rootNote}${octaveBase - 1} ~ ${fifthNote}${octaveBase - 1} ~`
    : `${rootNote}${octaveBase - 1}*2 ${fifthNote}${octaveBase - 1}*2`;

  return `note("${pattern}").s("sawtooth").lpf(400).decay(0.2).sustain(0.1)`;
}

/**
 * Generate full Strudel pattern from epistemic state
 */
export function generatePattern(vectors, options = {}) {
  const params = mapVectorsToParams(vectors);
  const { includeComments = true, mode = 'full' } = options;

  let code = '';

  if (includeComments) {
    code += `// Epistemic DJ - Generated Pattern\n`;
    code += `// know: ${vectors.know?.toFixed(2) || '?'} | uncertainty: ${vectors.uncertainty?.toFixed(2) || '?'}\n`;
    code += `// engagement: ${vectors.engagement?.toFixed(2) || '?'} | clarity: ${vectors.clarity?.toFixed(2) || '?'}\n`;
    code += `// tempo: ${params.tempo} BPM | scale: ${params.scale.join('-')}\n\n`;
  }

  // Set tempo
  code += `setcps(${(params.tempo / 60 / 4).toFixed(3)})\n\n`;

  if (mode === 'minimal') {
    // Just melody
    code += generateMelody(params);
    code += `\n  .s("triangle")`;
    code += `\n  .lpf(${params.cutoff})`;
    code += `\n  .gain(${params.gain.toFixed(2)})`;
  } else if (mode === 'drums') {
    // Just drums
    code += generateDrums(params);
  } else {
    // Full pattern with all elements
    code += `stack(\n`;
    code += `  // Melody\n`;
    code += `  ${generateMelody(params)}\n`;
    code += `    .s("triangle")\n`;
    code += `    .lpf(${params.cutoff})\n`;
    code += `    .room(${params.roomSize.toFixed(2)})\n`;
    code += `    .gain(${(params.gain * 0.6).toFixed(2)}),\n\n`;

    code += `  // Bass\n`;
    code += `  ${generateBass(params)}\n`;
    code += `    .gain(${(params.gain * 0.7).toFixed(2)}),\n\n`;

    code += `  // Drums\n`;
    code += `  ${generateDrums(params)}\n`;
    code += `)`;
  }

  return code;
}

/**
 * Generate pattern for specific moods
 */
export function generateMoodPattern(mood) {
  const moods = {
    focus: {
      know: 0.6, uncertainty: 0.3, engagement: 0.5, clarity: 0.7,
      coherence: 0.8, signal: 0.4, density: 0.3, state: 0.5,
      change: 0.2, completion: 0.3, impact: 0.5, context: 0.5, do: 0.6
    },
    energize: {
      know: 0.7, uncertainty: 0.2, engagement: 0.9, clarity: 0.8,
      coherence: 0.7, signal: 0.7, density: 0.6, state: 0.6,
      change: 0.4, completion: 0.5, impact: 0.8, context: 0.6, do: 0.8
    },
    reflect: {
      know: 0.4, uncertainty: 0.5, engagement: 0.3, clarity: 0.4,
      coherence: 0.6, signal: 0.3, density: 0.2, state: 0.4,
      change: 0.3, completion: 0.2, impact: 0.4, context: 0.7, do: 0.3
    },
    debug: {
      know: 0.5, uncertainty: 0.6, engagement: 0.7, clarity: 0.5,
      coherence: 0.5, signal: 0.6, density: 0.5, state: 0.4,
      change: 0.5, completion: 0.3, impact: 0.6, context: 0.5, do: 0.6
    },
    celebrate: {
      know: 0.9, uncertainty: 0.1, engagement: 0.95, clarity: 0.9,
      coherence: 0.9, signal: 0.8, density: 0.7, state: 0.8,
      change: 0.3, completion: 0.95, impact: 0.9, context: 0.7, do: 0.9
    },
  };

  const vectors = moods[mood] || moods.focus;
  return generatePattern(vectors);
}

/**
 * Interpolate between two epistemic states
 */
export function interpolateStates(from, to, t) {
  const result = {};
  const keys = new Set([...Object.keys(from), ...Object.keys(to)]);

  for (const key of keys) {
    const fromVal = from[key] ?? 0.5;
    const toVal = to[key] ?? 0.5;
    result[key] = fromVal + (toVal - fromVal) * t;
  }

  return result;
}

// CLI usage
if (process.argv[1]?.endsWith('epistemic-to-strudel.js')) {
  const args = process.argv.slice(2);

  if (args[0] === '--mood') {
    console.log(generateMoodPattern(args[1] || 'focus'));
  } else if (args[0] === '--json') {
    const vectors = JSON.parse(args[1] || '{}');
    console.log(generatePattern(vectors));
  } else {
    // Demo with random vectors
    const demo = {
      know: Math.random(),
      uncertainty: Math.random(),
      engagement: Math.random(),
      clarity: Math.random(),
      coherence: Math.random(),
      signal: Math.random(),
      density: Math.random(),
      state: Math.random(),
      change: Math.random(),
      completion: Math.random(),
      impact: Math.random(),
      context: Math.random(),
      do: Math.random(),
    };
    console.log(generatePattern(demo));
  }
}
