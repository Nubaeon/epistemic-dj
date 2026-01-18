#!/usr/bin/env node
/**
 * Epistemic DJ - MCP Server
 * Exposes tools for generating music from epistemic state
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import { generatePattern, generateMoodPattern, interpolateStates } from '../generator/epistemic-to-strudel.js';

// Current state for real-time updates
let currentState = {
  vectors: null,
  pattern: null,
  isPlaying: false,
  mode: 'full',
};

// WebSocket connections for browser bridge (future)
const wsConnections = new Set();

/**
 * Tool definitions
 */
const TOOLS = [
  {
    name: 'generate_pattern',
    description: 'Generate a Strudel music pattern from epistemic vectors. Returns code that can be pasted into strudel.cc',
    inputSchema: {
      type: 'object',
      properties: {
        vectors: {
          type: 'object',
          description: 'Epistemic state vectors (0-1 scale)',
          properties: {
            know: { type: 'number', description: 'Knowledge/understanding level' },
            uncertainty: { type: 'number', description: 'Uncertainty level' },
            engagement: { type: 'number', description: 'Engagement/focus level' },
            clarity: { type: 'number', description: 'Clarity of understanding' },
            coherence: { type: 'number', description: 'Coherence of knowledge' },
            signal: { type: 'number', description: 'Signal strength/density' },
            density: { type: 'number', description: 'Information density' },
            state: { type: 'number', description: 'Current state assessment' },
            change: { type: 'number', description: 'Rate of change' },
            completion: { type: 'number', description: 'Task completion level' },
            impact: { type: 'number', description: 'Perceived impact' },
            context: { type: 'number', description: 'Context understanding' },
            do: { type: 'number', description: 'Capability/ability level' },
          },
        },
        mode: {
          type: 'string',
          enum: ['full', 'minimal', 'drums'],
          description: 'Pattern complexity mode',
          default: 'full',
        },
        includeComments: {
          type: 'boolean',
          description: 'Include explanatory comments in output',
          default: true,
        },
      },
      required: ['vectors'],
    },
  },
  {
    name: 'generate_mood',
    description: 'Generate a pattern for a specific mood preset',
    inputSchema: {
      type: 'object',
      properties: {
        mood: {
          type: 'string',
          enum: ['focus', 'energize', 'reflect', 'debug', 'celebrate'],
          description: 'Mood preset to generate',
        },
      },
      required: ['mood'],
    },
  },
  {
    name: 'get_pattern_url',
    description: 'Get a shareable URL for the current pattern on strudel.cc',
    inputSchema: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          description: 'Strudel pattern code (optional, uses last generated if not provided)',
        },
      },
    },
  },
  {
    name: 'explain_mapping',
    description: 'Explain how epistemic vectors map to musical parameters',
    inputSchema: {
      type: 'object',
      properties: {
        vector: {
          type: 'string',
          description: 'Specific vector to explain (optional, explains all if not provided)',
        },
      },
    },
  },
  {
    name: 'crossfade_pattern',
    description: 'Generate a series of patterns that transition between two epistemic states',
    inputSchema: {
      type: 'object',
      properties: {
        from: {
          type: 'object',
          description: 'Starting epistemic state',
        },
        to: {
          type: 'object',
          description: 'Target epistemic state',
        },
        steps: {
          type: 'number',
          description: 'Number of intermediate patterns',
          default: 4,
        },
      },
      required: ['from', 'to'],
    },
  },
];

/**
 * Vector to music mapping explanations
 */
const MAPPING_EXPLANATIONS = {
  know: {
    musical: 'Scale complexity and consonance',
    low: 'Dissonant, uncertain scales (diminished)',
    high: 'Consonant, confident scales (pentatonic, major)',
  },
  uncertainty: {
    musical: 'Pattern degradation and probability',
    low: 'Solid, predictable patterns',
    high: 'Random note drops, unpredictable rhythms',
  },
  engagement: {
    musical: 'Tempo (60-140 BPM) and drum intensity',
    low: 'Slow, sparse, contemplative',
    high: 'Fast, driving, energetic',
  },
  clarity: {
    musical: 'Filter cutoff frequency (brightness)',
    low: 'Dark, muffled, low-pass filtered',
    high: 'Bright, present, clear',
  },
  coherence: {
    musical: 'Rhythmic stability (inverse)',
    low: 'Complex, syncopated rhythms',
    high: 'Steady, predictable beats',
  },
  signal: {
    musical: 'Hi-hat density and note count',
    low: 'Sparse, minimal notes',
    high: 'Dense, busy patterns',
  },
  density: {
    musical: 'Octave range and layering',
    low: 'Narrow range, simple',
    high: 'Wide range, complex layers',
  },
  state: {
    musical: 'Reverb/room size (space)',
    low: 'Dry, close, intimate',
    high: 'Spacious, atmospheric',
  },
  change: {
    musical: 'Pattern variation (jux, rev)',
    low: 'Static, repetitive',
    high: 'Evolving, varied',
  },
  completion: {
    musical: 'Build-up intensity',
    low: 'Sparse, beginning feel',
    high: 'Full, climactic',
  },
  impact: {
    musical: 'Overall volume/gain',
    low: 'Quiet, subtle',
    high: 'Loud, powerful',
  },
  context: {
    musical: 'Base octave (pitch register)',
    low: 'Higher register',
    high: 'Lower register, grounded',
  },
  do: {
    musical: 'Combined with know for scale selection',
    low: 'Tentative scales',
    high: 'Confident, resolving scales',
  },
};

/**
 * Handle tool calls
 */
async function handleToolCall(name, args) {
  switch (name) {
    case 'generate_pattern': {
      const { vectors, mode = 'full', includeComments = true } = args;
      const pattern = generatePattern(vectors, { mode, includeComments });
      currentState.vectors = vectors;
      currentState.pattern = pattern;
      currentState.mode = mode;
      return {
        content: [
          {
            type: 'text',
            text: pattern,
          },
        ],
        metadata: {
          tempo: Math.round(60 + ((vectors.engagement || 0.5) * 80)),
          mode,
        },
      };
    }

    case 'generate_mood': {
      const { mood } = args;
      const pattern = generateMoodPattern(mood);
      currentState.pattern = pattern;
      return {
        content: [
          {
            type: 'text',
            text: pattern,
          },
        ],
        metadata: { mood },
      };
    }

    case 'get_pattern_url': {
      const pattern = args.pattern || currentState.pattern;
      if (!pattern) {
        return {
          content: [{ type: 'text', text: 'No pattern available. Generate one first.' }],
          isError: true,
        };
      }
      // Encode pattern for URL
      const encoded = encodeURIComponent(pattern);
      const url = `https://strudel.cc/#${btoa(pattern)}`;
      return {
        content: [
          {
            type: 'text',
            text: `Open in Strudel REPL:\n${url}\n\nOr copy the pattern and paste into strudel.cc`,
          },
        ],
      };
    }

    case 'explain_mapping': {
      const { vector } = args;
      if (vector && MAPPING_EXPLANATIONS[vector]) {
        const exp = MAPPING_EXPLANATIONS[vector];
        return {
          content: [
            {
              type: 'text',
              text: `**${vector}** → ${exp.musical}\n- Low (0.0): ${exp.low}\n- High (1.0): ${exp.high}`,
            },
          ],
        };
      }
      // Explain all
      let text = '# Epistemic → Musical Mappings\n\n';
      for (const [vec, exp] of Object.entries(MAPPING_EXPLANATIONS)) {
        text += `**${vec}** → ${exp.musical}\n`;
        text += `  - Low: ${exp.low}\n`;
        text += `  - High: ${exp.high}\n\n`;
      }
      return { content: [{ type: 'text', text }] };
    }

    case 'crossfade_pattern': {
      const { from, to, steps = 4 } = args;
      const patterns = [];
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const interpolated = interpolateStates(from, to, t);
        patterns.push({
          step: i,
          t: t.toFixed(2),
          pattern: generatePattern(interpolated, { includeComments: false }),
        });
      }
      return {
        content: [
          {
            type: 'text',
            text: patterns.map(p => `// Step ${p.step} (t=${p.t})\n${p.pattern}`).join('\n\n---\n\n'),
          },
        ],
      };
    }

    default:
      return {
        content: [{ type: 'text', text: `Unknown tool: ${name}` }],
        isError: true,
      };
  }
}

/**
 * Main server setup
 */
async function main() {
  const server = new Server(
    {
      name: 'epistemic-dj',
      version: '0.1.0',
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // List tools
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS,
  }));

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    return handleToolCall(name, args || {});
  });

  // Connect via stdio
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error('Epistemic DJ MCP Server running on stdio');
}

main().catch(console.error);
