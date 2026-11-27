#!/usr/bin/env node
/**
 * Render docs/media/demo.gif.
 *
 * This is an OFFLINE replay of the kinematic model implemented in
 * rl/envs/drone_swarm_env.py (same dt, max speed, max yaw rate, goal tolerance,
 * area and dynamics), driven by a scripted goal-seeking controller.
 *
 * It is NOT a Gazebo capture and NOT a trained policy: it exists so that the
 * observation/action contract and the task are visible without installing ROS 2.
 *
 * Pure Node, no dependencies: the frames are drawn into an indexed buffer and
 * written as an animated GIF89a (own LZW encoder), then decoded again as a
 * self-check before the file is written.
 *
 *     node scripts/render_demo.mjs [outfile]
 */

import { Buffer } from "node:buffer";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

// ----------------------------------------------------------------- parameters
const W = 400;
const H = 300;
const AREA = 10.0; // metres, same as EnvConfig.area
const MARGIN = 22;
const SCALE = Math.min((W - 2 * MARGIN) / AREA, (H - 2 * MARGIN) / AREA);

const N_AGENTS = 5;
const N_OBSTACLES = 5;
const DT = 0.1; // EnvConfig.dt
const MAX_V = 1.0; // EnvConfig.max_speed
const MAX_W = 1.5; // EnvConfig.max_yaw_rate
const TOL = 0.35; // EnvConfig.goal_tolerance
const FRAMES = 120;
const DELAY_CS = 7; // 0.07 s per frame -> ~14 fps

// ---------------------------------------------------------------- colour table
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const BG = [248, 250, 252];
const AGENT_RGB = [
  [248, 113, 113],
  [96, 165, 250],
  [74, 222, 128],
  [251, 191, 36],
  [192, 132, 252],
];

const PALETTE = [
  BG, // 0 background
  [226, 232, 240], // 1 grid
  [148, 163, 184], // 2 border
  [100, 116, 139], // 3 obstacle fill
  [71, 85, 105], // 4 obstacle edge
  ...AGENT_RGB, // 5..9 agents
  ...AGENT_RGB.map((c) => mix(c, BG, 0.55)), // 10..14 goal rings
  ...AGENT_RGB.map((c) => mix(c, BG, 0.3)), // 15..19 trails
];
while (PALETTE.length < 64) PALETTE.push(BG);
const AGENT = (i) => 5 + i;
const GOAL = (i) => 10 + i;
const TRAIL = (i) => 15 + i;

// ------------------------------------------------------------------- drawing
const buf = new Uint8Array(W * H);
const px = (x) => Math.round(MARGIN + x * SCALE);
const py = (y) => Math.round(H - MARGIN - y * SCALE);

function discWorld(x, y, r, idx) {
  const cx = px(x);
  const cy = py(y);
  const rad = Math.max(1, r * SCALE);
  const r2 = rad * rad;
  const x0 = Math.max(0, Math.floor(cx - rad));
  const x1 = Math.min(W - 1, Math.ceil(cx + rad));
  const y0 = Math.max(0, Math.floor(cy - rad));
  const y1 = Math.min(H - 1, Math.ceil(cy + rad));
  for (let yy = y0; yy <= y1; yy++) {
    for (let xx = x0; xx <= x1; xx++) {
      const dx = xx - cx;
      const dy = yy - cy;
      if (dx * dx + dy * dy <= r2) buf[yy * W + xx] = idx;
    }
  }
}

function ringWorld(x, y, r, idx, thicknessPx = 2) {
  discWorld(x, y, r, idx);
  const innerPx = Math.max(1, r * SCALE - thicknessPx);
  discWorld(x, y, innerPx / SCALE, 0);
}

function drawBackground() {
  buf.fill(0);
  for (let m = 0; m <= AREA; m += 1) {
    const gx = px(m);
    const gy = py(m);
    for (let y = py(0); y >= py(AREA); y--) buf[y * W + gx] = 1;
    for (let x = px(0); x <= px(AREA); x++) buf[gy * W + x] = 1;
  }
  for (let i = 0; i < 2; i++) {
    for (let x = px(0); x <= px(AREA); x++) {
      buf[(py(AREA) + i) * W + x] = 2;
      buf[(py(0) + i) * W + x] = 2;
    }
    for (let y = py(AREA); y <= py(0); y++) {
      buf[y * W + px(0) + i] = 2;
      buf[y * W + px(AREA) + i] = 2;
    }
  }
}

// ----------------------------------------------------------------- simulation
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rnd = mulberry32(20261127);
const rndRange = (lo, hi) => lo + (hi - lo) * rnd();

const obstacles = [];
while (obstacles.length < N_OBSTACLES) {
  const candidate = { x: rndRange(2.2, 7.8), y: rndRange(2.2, 7.8), r: rndRange(0.35, 0.7) };
  if (obstacles.every((o) => Math.hypot(o.x - candidate.x, o.y - candidate.y) > o.r + candidate.r + 1.3)) {
    obstacles.push(candidate);
  }
}

const freePoint = (minDist = 0.9) => {
  for (let tries = 0; tries < 500; tries++) {
    const p = { x: rndRange(1.0, AREA - 1.0), y: rndRange(1.0, AREA - 1.0) };
    const clearObstacles = obstacles.every((o) => Math.hypot(o.x - p.x, o.y - p.y) > o.r + minDist);
    if (clearObstacles) return p;
  }
  return { x: 5, y: 5 };
};

const agents = [];
for (let i = 0; i < N_AGENTS; i++) {
  let start = freePoint(1.2);
  for (let tries = 0; tries < 200 && agents.some((a) => Math.hypot(a.pos.x - start.x, a.pos.y - start.y) < 1.6); tries++) {
    start = freePoint(1.2);
  }
  const yaw = rndRange(-Math.PI, Math.PI);
  agents.push({
    pos: start,
    yaw,
    goal: freePoint(1.2),
    history: [],
  });
  for (let tries = 0; tries < 200 && Math.hypot(agents[i].goal.x - start.x, agents[i].goal.y - start.y) < 3.0; tries++) {
    agents[i].goal = freePoint(1.2);
  }
}

const wrap = (a) => {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
};

/** Scripted goal-seeking controller: heading to the goal, pushed away by the
 *  obstacles and the other agents. Not a learned policy. */
function controller(i) {
  const a = agents[i];
  let dx = a.goal.x - a.pos.x;
  let dy = a.goal.y - a.pos.y;
  const dist = Math.hypot(dx, dy) || 1e-6;
  dx /= dist;
  dy /= dist;

  for (const o of obstacles) {
    const ox = a.pos.x - o.x;
    const oy = a.pos.y - o.y;
    const d = Math.hypot(ox, oy) || 1e-6;
    if (d - o.r < 1.0) {
      const w = (1.0 - Math.max(0, d - o.r)) * 1.8;
      dx += (ox / d) * w;
      dy += (oy / d) * w;
    }
  }
  for (let j = 0; j < N_AGENTS; j++) {
    if (j === i) continue;
    const ox = a.pos.x - agents[j].pos.x;
    const oy = a.pos.y - agents[j].pos.y;
    const d = Math.hypot(ox, oy) || 1e-6;
    if (d < 0.9) {
      const w = (0.9 - d) * 1.6;
      dx += (ox / d) * w;
      dy += (oy / d) * w;
    }
  }

  const err = wrap(Math.atan2(dy, dx) - a.yaw);
  const wCmd = Math.max(-1, Math.min(1, err * 2.2));
  const speed = Math.max(0.1, Math.min(1, 1 - Math.abs(err) / 1.4)) * Math.min(1, dist / 0.8);
  return [speed, wCmd];
}

function simStep() {
  for (let i = 0; i < N_AGENTS; i++) {
    const a = agents[i];
    const [vCmd, wCmd] = controller(i);
    a.yaw = wrap(a.yaw + wCmd * MAX_W * DT);
    const v = vCmd * MAX_V;
    a.pos = {
      x: Math.min(AREA, Math.max(0, a.pos.x + Math.cos(a.yaw) * v * DT)),
      y: Math.min(AREA, Math.max(0, a.pos.y + Math.sin(a.yaw) * v * DT)),
    };
    a.history.push({ ...a.pos });
    if (a.history.length > 40) a.history.shift();
  }
}

function renderFrame(step) {
  drawBackground();

  for (const o of obstacles) {
    discWorld(o.x, o.y, o.r + 0.06, 4);
    discWorld(o.x, o.y, o.r, 3);
  }

  for (let i = 0; i < N_AGENTS; i++) {
    const a = agents[i];
    ringWorld(a.goal.x, a.goal.y, TOL, GOAL(i), 2);
    for (let k = 0; k < a.history.length; k += 4) {
      discWorld(a.history[k].x, a.history[k].y, 0.06, TRAIL(i));
    }
  }

  for (let i = 0; i < N_AGENTS; i++) {
    const a = agents[i];
    discWorld(a.pos.x, a.pos.y, 0.2, AGENT(i));
    discWorld(a.pos.x + Math.cos(a.yaw) * 0.28, a.pos.y + Math.sin(a.yaw) * 0.28, 0.08, AGENT(i));
    void step;
  }
  return buf.slice();
}

// -------------------------------------------------------------------- GIF LZW
function lzwEncode(indices, minCodeSize) {
  const clearCode = 1 << minCodeSize;
  const eoiCode = clearCode + 1;
  let codeSize = minCodeSize + 1;
  let next = eoiCode + 1;
  const dict = new Map();
  const bytes = [];
  let acc = 0;
  let accBits = 0;

  const emit = (code) => {
    acc |= code << accBits;
    accBits += codeSize;
    while (accBits >= 8) {
      bytes.push(acc & 0xff);
      acc >>= 8;
      accBits -= 8;
    }
    // Same convention as the reference encoder (Poskanzer): the code is written
    // at the current width, and the width is widened for the *following* code.
    if (next > (1 << codeSize) - 1 && codeSize < 12) codeSize++;
  };

  emit(clearCode);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const c = indices[i];
    const key = (prefix << 8) | c;
    const known = dict.get(key);
    if (known !== undefined) {
      prefix = known;
      continue;
    }
    emit(prefix);
    if (next < 4096) {
      dict.set(key, next++);
    } else {
      emit(clearCode);
      dict.clear();
      next = eoiCode + 1;
      codeSize = minCodeSize + 1;
    }
    prefix = c;
  }
  emit(prefix);
  emit(eoiCode);
  if (accBits > 0) bytes.push(acc & 0xff);
  return bytes;
}

function subBlocks(bytes) {
  const out = [];
  for (let i = 0; i < bytes.length; i += 255) {
    const chunk = bytes.slice(i, i + 255);
    out.push(chunk.length, ...chunk);
  }
  out.push(0);
  return out;
}

function buildGif(frames, minCodeSize, delayCs) {
  const gctBits = Math.max(1, Math.ceil(Math.log2(PALETTE.length)));
  const tableSize = 1 << gctBits;
  const gct = Buffer.alloc(tableSize * 3);
  PALETTE.forEach((c, i) => {
    gct[i * 3] = c[0];
    gct[i * 3 + 1] = c[1];
    gct[i * 3 + 2] = c[2];
  });

  const out = [];
  out.push(Buffer.from("GIF89a", "ascii"));

  const lsd = Buffer.alloc(7);
  lsd.writeUInt16LE(W, 0);
  lsd.writeUInt16LE(H, 2);
  lsd[4] = 0x80 | 0x70 | (gctBits - 1); // global table, 8-bit colour resolution
  lsd[5] = 0;
  lsd[6] = 0;
  out.push(lsd, gct);

  // NETSCAPE2.0 loop forever
  out.push(
    Buffer.from([0x21, 0xff, 0x0b, ...Buffer.from("NETSCAPE2.0", "ascii"), 0x03, 0x01, 0x00, 0x00, 0x00]),
  );

  for (const frame of frames) {
    const gce = Buffer.alloc(8);
    gce[0] = 0x21;
    gce[1] = 0xf9;
    gce[2] = 0x04;
    gce[3] = 0x00;
    gce.writeUInt16LE(delayCs, 4);
    gce[7] = 0x00;
    out.push(gce);

    const desc = Buffer.alloc(10);
    desc[0] = 0x2c;
    desc.writeUInt16LE(0, 1);
    desc.writeUInt16LE(0, 3);
    desc.writeUInt16LE(W, 5);
    desc.writeUInt16LE(H, 7);
    desc[9] = 0x00;
    out.push(desc);

    out.push(Buffer.from([minCodeSize, ...subBlocks(lzwEncode(frame, minCodeSize))]));
  }

  out.push(Buffer.from([0x3b]));
  return Buffer.concat(out);
}

// ------------------------------------------------------------------- decoding
/** Minimal GIF/LZW decoder used only to prove the file we wrote is readable. */
function lzwDecode(data, minCodeSize, expectedLength) {
  const clearCode = 1 << minCodeSize;
  const eoiCode = clearCode + 1;
  let dict = [];
  let codeSize = minCodeSize + 1;
  const reset = () => {
    dict = [];
    for (let i = 0; i < clearCode; i++) dict.push([i]);
    dict.push(null);
    dict.push(null);
    codeSize = minCodeSize + 1;
  };
  reset();

  const out = [];
  let bitPos = 0;
  let prev = null;
  const readCode = () => {
    let code = 0;
    for (let i = 0; i < codeSize; i++) {
      const byte = data[bitPos >> 3];
      if (byte === undefined) return -1;
      code |= ((byte >> (bitPos & 7)) & 1) << i;
      bitPos++;
    }
    return code;
  };

  for (;;) {
    const code = readCode();
    if (code === -1 || code === eoiCode) break;
    if (code === clearCode) {
      reset();
      prev = null;
      continue;
    }
    let entry;
    if (code < dict.length && dict[code]) entry = dict[code];
    else if (prev) entry = prev.concat(prev[0]);
    else throw new Error(`undecodable code ${code}`);

    for (const v of entry) out.push(v);
    if (prev) {
      dict.push(prev.concat(entry[0]));
      if (dict.length === 1 << codeSize && codeSize < 12) codeSize++;
    }
    prev = entry;
  }
  if (expectedLength !== undefined && out.length !== expectedLength) {
    throw new Error(`decoded ${out.length} pixels, expected ${expectedLength}`);
  }
  return out;
}

// ----------------------------------------------------------------------- main
const outFile = process.argv[2] ?? "docs/media/demo.gif";
const minCodeSize = Math.max(2, Math.ceil(Math.log2(PALETTE.length)));

const frames = [];
for (let f = 0; f < FRAMES; f++) {
  simStep();
  frames.push(renderFrame(f));
}

// self-check: decode every frame back and compare with what was drawn
const gif = buildGif(frames, minCodeSize, DELAY_CS);
let checked = 0;
{
  let p = 13; // header + LSD
  const gctBits = Math.max(1, Math.ceil(Math.log2(PALETTE.length)));
  p += 3 * (1 << gctBits); // global colour table
  while (p < gif.length) {
    const marker = gif[p];
    if (marker === 0x21) {
      const label = gif[p + 1];
      p += 2;
      let size = gif[p++];
      while (size !== 0) {
        p += size;
        size = gif[p++];
      }
      void label;
    } else if (marker === 0x2c) {
      p += 10;
      const lzwMin = gif[p++];
      const bytes = [];
      let size = gif[p++];
      while (size !== 0) {
        for (let i = 0; i < size; i++) bytes.push(gif[p + i]);
        p += size;
        size = gif[p++];
      }
      const decoded = lzwDecode(bytes, lzwMin, W * H);
      const original = frames[checked];
      for (let i = 0; i < W * H; i++) {
        if (decoded[i] !== original[i]) {
          throw new Error(`frame ${checked}: pixel ${i} decoded as ${decoded[i]}, drew ${original[i]}`);
        }
      }
      checked++;
    } else if (marker === 0x3b) {
      break;
    } else {
      throw new Error(`unexpected block 0x${marker.toString(16)} at ${p}`);
    }
  }
}

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, gif);

const reached = agents.filter((a) => Math.hypot(a.goal.x - a.pos.x, a.goal.y - a.pos.y) < TOL).length;

console.log(
  `${outFile}: ${checked}/${FRAMES} frames verified, ${W}x${H}, ${(gif.length / 1024).toFixed(0)} KiB, ` +
    `${reached}/${N_AGENTS} agents on goal at the end`,
);
