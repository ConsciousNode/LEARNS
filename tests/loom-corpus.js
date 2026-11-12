#!/usr/bin/env node
/**
 * LEARNS v0.0016 "Rhythm" — Procedural .loom Corpus Generator, AuraLoom Validator,
 * Byte-Role Classifier, and Crew Leak Checker (tests/loom-corpus.js)
 *
 * PROVENANCE:
 * 1. Composition Engine lifted from Synth Radio v13+ (ConsciousNode SoftWorks,
 *    synth-radio/index.html lines 453-744 and 3464-4903):
 *    - mulberry32 PRNG, SCALES, CHORD_PROGRESSIONS, Euclidean rhythms (EUCLID),
 *      DRUM_PATTERNS, all 146 seeded CHANNELS across 12 genre clusters,
 *      getChannelStyle, CLUSTER_PALETTE, buildVoiceRoles, and generateSong.
 *    - Audio synthesis (WebAudio DSP nodes) omitted; music structure, harmony,
 *      voice selection, and rhythms preserved 100% intact.
 * 2. Score Parser & Loom Renderer lifted from AuraLoom v0.4/v0.5 (ConsciousNode
 *    SoftWorks, auraloom/index.html lines 304-349, 1201-1231, 1881-2003):
 *    - parseNote, parseWhere, SCALES, TIMBRES, FM, besselJ, partialsAt,
 *      stampNote, pluck, kick, snare, hat, put, rub, layout, weaveScore.
 */
'use strict';

const fs = require('fs');
const path = require('path');

/* ════════════════════════════════════════════════════════════════
   1. SYNTH RADIO -> AURALOOM VOICE & SCALE MAPPING
   ════════════════════════════════════════════════════════════════ */

/**
 * Maps all 50 Synth Radio voices to the nearest of AuraLoom's 12 voices:
 *   Additive TIMBRES: pure, flute, string, reed, organ, bell
 *   Genesis 2-op FM:  fmLead, fmBass, fmKeys, fmBrass, fmBell, fmClang
 */
const SYNTH_RADIO_TO_AURALOOM_VOICE = {
  // Keys / Pianos / Electric Pianos
  piano_acoustic:    'fmKeys',
  piano_chop:        'fmKeys',
  piano_electric:    'fmKeys',
  wurlitzer:         'fmKeys',
  fm_keys:           'fmKeys',
  clavinet:          'fmKeys',
  harpsichord:       'string',
  // Organs / Accordions / Choir
  organ_hammond:     'organ',
  organ_pipe:        'organ',
  organ_gospel:      'organ',
  accordion:         'organ',
  choir_aaah:        'organ',
  // Mallets / Bells / Metallic
  vibraphone:        'bell',
  marimba:           'bell',
  kalimba:           'bell',
  celesta:           'bell',
  music_box:         'bell',
  synth_bell:        'fmBell',
  ym2612_bell:       'fmBell',
  church_bell:       'fmClang',
  cowbell:           'fmClang',
  // Strings / Plucked / Guitar
  guitar_clean:      'string',
  guitar_muted:      'string',
  guitar_distorted:  'fmLead',
  strings_section:   'string',
  pizz_strings:      'string',
  sitar_drone:       'string',
  koto:              'string',
  banjo:             'string',
  dulcimer:          'string',
  // Basses
  bass_acoustic:     'fmBass',
  bass_electric:     'fmBass',
  bass_808:          'fmBass',
  bass_moog:         'fmBass',
  bass_wobble:       'fmBass',
  acid_bass:         'fmBass',
  reese_bass:        'fmBass',
  ym2612_bass:       'fmBass',
  // Winds / Reeds / Brass
  flute_breathy:     'flute',
  ocarina:           'flute',
  pan_flute:         'flute',
  sax:               'reed',
  harmonica:         'reed',
  brass_section:     'fmBrass',
  synth_brass:       'fmBrass',
  // Synth Leads / Pads / Chiptune / Pure
  synth_lead_saw:    'fmLead',
  synth_lead_square: 'fmLead',
  supersaw:          'fmLead',
  saw_pluck:         'fmLead',
  synth_arp:         'fmLead',
  synth_pluck:       'fmKeys',
  ym2612:            'fmLead',
  ym2612_lead:       'fmLead',
  chiptune_pulse:    'reed',
  chiptune_saw:      'fmLead',
  chiptune_triangle: 'pure',
  theremin:          'pure',
  synth_pad_warm:    'organ',
  synth_pad_analog:  'string',
  synth_pad_glass:   'flute',
  ambient_pad:       'pure'
};

// VOICES lookup object expected by Synth Radio's generateSong & buildVoiceRoles
const VOICES = Object.assign({}, SYNTH_RADIO_TO_AURALOOM_VOICE);

// Synth Radio scale name -> AuraLoom scale name
const SYNTH_RADIO_TO_AURALOOM_SCALE = {
  major:            'major',
  minor:            'minor',
  dorian:           'dorian',
  phrygian:         'phrygian',
  lydian:           'lydian',
  mixolydian:       'mixolydian',
  locrian:          'locrian',
  harmonic_minor:   'harmonicMinor',
  blues:            'blues',
  minor_pentatonic: 'pentMin',
  major_pentatonic: 'pentMaj',
  melodic_minor:    'harmonicMinor',
  hungarian_minor:  'harmonicMinor',
  maqam_hijaz:      'harmonicMinor',
  raga_bhairav:     'harmonicMinor',
  japanese_hirajoshi: 'phrygian',
  raga_bhairavi:    'phrygian',
  maqam_rast:       'pentMaj',
  gamelan_slendro:  'pentMaj',
  gamelan_pelog:    'pentMin',
  whole_tone:       'chromatic',
  diminished:       'chromatic',
  chromatic:        'chromatic',
  bohlen_pierce:    'chromatic'
};

const SYNTH_RADIO_TO_AURALOOM_DRUM = {
  kick:             'kick',
  kick_808:         'kick',
  kick_909:         'kick',
  kick_acoustic:    'kick',
  tom:              'kick',
  tom_high:         'kick',
  tom_low:          'kick',
  djembe:           'kick',
  tabla:            'kick',
  snare:            'snare',
  snare_909:        'snare',
  snare_electronic: 'snare',
  clap:             'snare',
  rim:              'snare',
  snap:             'snare',
  clave:            'snare',
  cowbell:          'snare',
  hatc:             'hat',
  shaker:           'hat',
  ride:             'hat',
  perc:             'hat',
  hato:             'openhat',
  crash:            'openhat'
};

/* ════════════════════════════════════════════════════════════════
   2. LIFTED SYNTH RADIO COMPOSITION ENGINE (v13+)
   ════════════════════════════════════════════════════════════════ */

/* ─────────────────────────────────────────────
   PRNG (Mulberry32) — deterministic per seed
   ───────────────────────────────────────────── */
function mulberry32(seed){
  let s = seed >>> 0;
  return function(){
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/* helper: integer in [min,max] from a rng */
function rint(rng,min,max){ return Math.floor(rng()*(max-min+1))+min; }
function rfloat(rng,min,max){ return rng()*(max-min)+min; }
function pick(rng,arr){ return arr[Math.floor(rng()*arr.length)]; }
function chance(rng,p){ return rng()<p; }

/* ─────────────────────────────────────────────
   SCALES, NOTE->FREQ, CHORD PROGRESSIONS
   ───────────────────────────────────────────── */
const SCALES = {
  major:[0,2,4,5,7,9,11], minor:[0,2,3,5,7,8,10],
  minor_pentatonic:[0,3,5,7,10], major_pentatonic:[0,2,4,7,9],
  dorian:[0,2,3,5,7,9,10], phrygian:[0,1,3,5,7,8,10], mixolydian:[0,2,4,5,7,9,10],
  lydian:[0,2,4,6,7,9,11], locrian:[0,1,3,5,6,8,10],
  blues:[0,3,5,6,7,10], whole_tone:[0,2,4,6,8,10], diminished:[0,2,3,5,6,8,9,11],
  chromatic:[0,1,2,3,4,5,6,7,8,9,10,11],
  harmonic_minor:[0,2,3,5,7,8,11], melodic_minor:[0,2,3,5,7,9,11],
  maqam_rast:[0,2,4,5,7,8,10], maqam_hijaz:[0,1,4,5,7,8,10],
  raga_bhairav:[0,1,4,5,7,8,11], raga_bhairavi:[0,1,3,5,7,8,11],
  gamelan_slendro:[0,2,5,7,10], gamelan_pelog:[0,1,3,7,8],
  bohlen_pierce:[0,2,4,6,8,10,12], // approximation in semitones
  hungarian_minor:[0,2,3,6,7,8,11], japanese_hirajoshi:[0,2,3,7,8],
};
/* semitone offsets for chord qualities relative to a root degree (in scale degrees index) */
const CHORD_PROGRESSIONS = {
  "I-V-vi-IV":[0,7,9,5], "i-VII-VI-VII":[0,10,8,10], "i-iv-VII-III":[0,5,10,3],
  "ii-V-I":[2,7,0], "I-vi-IV-V":[0,9,5,7], "i-VI-III-VII":[0,8,3,10],
  "I-IV-V-I":[0,5,7,0], "vi-IV-I-V":[9,5,0,7], "i-v-VI-iv":[0,7,8,5],
  "I-V-vi-iii":[0,7,9,4], "ii-vi-IV-I":[2,9,5,0], "i-VII-VI-V":[0,10,8,7],
  "I-iii-IV-V":[0,4,5,7], "bVI-bVII-i":[8,10,0], "iv-bVII-i-v":[5,10,0,7],
  "I-bVII-IV-I":[0,10,5,0], "i-bVI-bVII-i":[0,8,10,0], "I-IV-vi-V":[0,5,9,7],
  // Japanese Royal Road (Ōdō) IV-V-iii-vi
  "IV-V-iii-vi":[5,7,4,9],
  // Sasanishiki IV-V-I-vi
  "IV-V-I-vi":[5,7,0,9],
  // Lo-fi / downtempo color
  "i-VII-VI-III":[0,10,8,3],
  // Jazz turnaround
  "ii-V-I-VI":[2,7,0,9],
  "I-VI-ii-V":[0,9,2,7],
  // Modal rock / metal
  "i-VI-bVII-i":[0,8,10,0],
  // Pop variant
  "vi-V-IV-V":[9,7,5,7],
};

const A4 = 440, MIDI_A4 = 69;
function midiToFreq(m){ return A4 * Math.pow(2,(m-MIDI_A4)/12); }
/* microtonal: allow fractional semitone for microtonal channels */
function midiToFreqMT(m){ return A4 * Math.pow(2,(m-MIDI_A4)/12); }

/* build a scale starting at a given root midi note, returns array of midi notes across ~2 octaves */
function buildScale(scaleName, rootMidi, octaves){
  octaves = octaves || 3;
  const steps = SCALES[scaleName] || SCALES.minor_pentatonic;
  const notes = [];
  for(let o=0;o<octaves;o++){
    for(let i=0;i<steps.length;i++){ notes.push(rootMidi + steps[i] + o*12); }
  }
  return notes;
}
/* scale degree -> midi note (degree can be fractional-ish index) */
function scaleNote(scaleArr, degree){
  const n = scaleArr.length;
  const idx = ((degree % n) + n) % n;
  const oct = Math.floor(degree / n);
  return scaleArr[idx] + oct*12;
}

/* ─────────────────────────────────────────────
   DRUM PATTERN LIBRARY (16-step, per beat 4 steps)
   each pattern: { kick:[...16], snare:[...16], ... } booleans
   ───────────────────────────────────────────── */
function emptyPattern(){ return {kick:[],snare:[],clap:[],hatc:[],hato:[],rim:[],snap:[],tom:[],perc:[],ride:[],crash:[],cowbell:[],shaker:[]}; }
function makePatterns(){
  const P = {};
  // 16-step helpers
  const s16 = (arr)=>{ const p=emptyPattern(); for(const k in arr) p[k]=arr[k]; return p; };
  const beat = (str)=>{ return str.replace(/[^x.]/g,'').split('').map(c=>c==='x'); };

  P.boom_bap = s16({
    kick:  beat("x...x...x...x..."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    hato:  beat("..............x."),
    perc:  beat("..x.....x.....x."),
  });
  P.lofi = s16({
    kick:  beat("x.....x.x......."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    perc:  beat("..............x."),
  });
  P.four_floor = s16({
    kick:  beat("x...x...x...x..."),
    clap:  beat("....x.......x..."),
    hatc:  beat("..x...x...x...x."),
    hato:  beat("..........x....."),
    perc:  beat("....x...x...x..."),
  });
  P.trap = s16({
    kick:  beat("x.....x...x....."),
    snare: beat("........x......."),
    hatc:  beat("x.xxx.xxx.xxx.xx"),
    hato:  beat("..............x."),
    perc:  beat("...x.......x...."),
  });
  P.dnb = s16({
    kick:  beat("x.....x...x....."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    perc:  beat("..x...x...x...x."),
  });
  P.techno = s16({
    kick:  beat("x...x...x...x..."),
    hatc:  beat("..x...x...x...x."),
    hato:  beat("....x...x...x...x".slice(0,16)),
    perc:  beat("x...x...x...x..."),
    clap:  beat("........x......."),
  });
  P.reggae = s16({
    kick:  beat("x.......x......."),
    hatc:  beat("..x...x...x...x."),
    hato:  beat("....x.......x..."),
    perc:  beat("..x...x...x...x."),
  });
  P.rock = s16({
    kick:  beat("x.......x...x..."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    crash: beat("x..............."),
  });
  P.metal = s16({
    kick:  beat("x.x.x.x.x.x.x.x."),
    snare: beat("....x.......x..."),
    hatc:  beat("xxxxxxxxxxxxxxxx"),
    crash: beat("x..............."),
  });
  P.dubstep = s16({
    kick:  beat("x.......x......."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    wobble:beat("x...x...x...x..."),
  });
  P.samba = s16({
    kick:  beat("x..x..x...x..x.."),
    snare: beat("..x...x...x...x."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    shaker:beat("xxxxxxxxxxxxxxxx"),
    perc:  beat("x.x.x.x.x.x.x.x."),
  });
  P.bossa = s16({
    kick:  beat("x.....x.x......."),
    hatc:  beat("..x...x...x...x."),
    perc:  beat("....x.......x..."),
  });
  P.afrobeat = s16({
    kick:  beat("x..x..x...x..x.."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    shaker:beat("xxxxxxxxxxxxxxxx"),
    perc:  beat("..x...x...x...x."),
  });
  P.waltz = s16({
    kick:  beat("x.......x......."),
    snare: beat("....x.......x..."),
    hatc:  beat("..x...x...x...x."),
  });
  P.breakbeat = s16({
    kick:  beat("x.....x...x....."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    perc:  beat("..x.....x.....x."),
  });
  P.silence = s16({});
  // J-pop / anime bright 8th-note drive
  P.jpop = s16({
    kick:  beat("x...x...x...x..."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    clap:  beat("....x.......x..."),
    hato:  beat("..........x....."),
  });
  // City pop / future funk swung groove
  P.city_pop = s16({
    kick:  beat("x.....x.x......."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    perc:  beat("..x.....x.....x."),
    clap:  beat("....x.......x..."),
  });
  // Shibuya-kei light bossa-pop
  P.shibuya = s16({
    kick:  beat("x.....x.x......."),
    hatc:  beat("..x...x...x...x."),
    snare: beat("....x.......x..."),
    shaker:beat("x.x.x.x.x.x.x.x."),
  });
  // Future funk chopped
  P.future_funk = s16({
    kick:  beat("x...x.x.x...x..."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.xxx.x.x.xxx.x."),
    clap:  beat("....x.......x..."),
  });
  // Synthpop / anime OP
  P.synthpop = s16({
    kick:  beat("x...x...x...x..."),
    snare: beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    clap:  beat("....x.......x..."),
  });
  // House variant with more offbeat hats
  P.house_swing = s16({
    kick:  beat("x...x...x...x..."),
    clap:  beat("....x.......x..."),
    hatc:  beat("x.x.x.x.x.x.x.x."),
    hato:  beat("..x...x...x...x."),
  });
  return P;
}

/* Bjorklund Euclidean rhythm E(k,n) — most even distribution of k hits in n steps */
function euclidean(k, n){
  k = Math.max(0, Math.min(k|0, n|0));
  n = Math.max(1, n|0);
  if(k===0) return Array(n).fill(false);
  if(k===n) return Array(n).fill(true);
  const pattern = [];
  const counts = [];
  const remainders = [];
  let divisor = n - k;
  remainders.push(k);
  let level = 0;
  while(remainders[level] > 1){
    counts.push(Math.floor(divisor / remainders[level]));
    remainders.push(divisor % remainders[level]);
    divisor = remainders[level];
    level++;
  }
  counts.push(divisor);
  function build(l){
    if(l === -1){ pattern.push(false); return; }
    if(l === -2){ pattern.push(true); return; }
    for(let i=0;i<counts[l];i++) build(l-1);
    if(remainders[l] !== 0) build(l-2);
  }
  build(level);
  // rotate so first hit is on step 0 when possible
  const idx = pattern.indexOf(true);
  if(idx > 0){
    const rot = pattern.slice(idx).concat(pattern.slice(0, idx));
    return rot.slice(0, n);
  }
  return pattern.slice(0, n);
}
// Named Euclidean templates used by genre styles
const EUCLID = {
  tresillo: () => euclidean(3, 8),      // x..x..x.
  cinquillo: () => euclidean(5, 8),     // x.xx.xx.
  bossa: () => euclidean(5, 16),        // x..x..x..x..x...
  afro: () => euclidean(7, 16),
  clave: () => euclidean(3, 16).map((v,i)=> v || (i===6||i===12)), // approx son clave-ish overlay
  four: () => euclidean(4, 16),
  six: () => euclidean(6, 16),
};


const DRUM_PATTERNS = makePatterns();

/* ─────────────────────────────────────────────
   BREAK TYPES
   ───────────────────────────────────────────── */
const BREAK_TYPES = ["jingle","interlude","sfx","tone","station_id"];
const BREAK_NAMES = { jingle:"Station Jingle", interlude:"Musical Interlude", sfx:"SFX Transition", tone:"Tone Break", station_id:"Station ID" };

/* ════════════════════════════════════════════════════════════════
   VOICE LIBRARY  —  40+ synthesizer voice functions
   Each voice: function(ctx, freq, time, dur, bus, params)
   params may include: { vel, gain, decay, cutoff, detune, reverbSend, ... }

/* ════════════════════════════════════════════════════════════════
   CHANNEL LIBRARY — all 146 channels across 12 clusters
   ════════════════════════════════════════════════════════════════ */
function ch(id,name,cluster,opts){
  opts = opts||{};
  return Object.assign({
    id, name, cluster,
    bpm:[90,110], swing:[0,0], scale:'minor', key:null, timeSig:4,
    structure:['intro','verse','chorus','verse','chorus','bridge','outro'],
    sectionBars:{intro:4,verse:8,prechorus:4,chorus:8,bridge:4,outro:4},
    drums:{pattern:'four_floor', kick:true,snare:true,hihat:true,rimshot:false,snap:false,clap:false},
    voices:['piano_chop','bass_808','wurlitzer','guitar_muted','synth_pad_warm'], leadVoice:'piano_chop',
    density:0.4, reverb:0.4, filters:{}, tempo_humanise:0.02,
    chord_progression:null, dynamics:{verse:0.7,prechorus:0.85,chorus:0.95,bridge:0.55,intro:0.55,outro:0.5},
    mastering:{}, texture:null, beatless:false, special:null,
  }, opts);
}

const CHANNELS = [
/* ── ELECTRONIC (25) ── */
ch('lofi_hip_hop','Lo-Fi Hip Hop','electronic',{bpm:[76,88],scale:'minor_pentatonic',swing:[0.58,0.66],drums:{pattern:'lofi',kick:true,snare:true,hihat:true,rimshot:true},voices:['piano_chop','bass_electric','wurlitzer','vinyl_crackle'],leadVoice:'piano_chop',density:0.32,reverb:0.62,filters:{master_lp:7200},tempo_humanise:0.04,chord_progression:'i-VII-VI-III',structure:['intro','verse','chorus','verse','chorus','outro'],sectionBars:{intro:4,verse:8,chorus:8,outro:4},mastering:{eqLow:2.5,eqMid:-1.5,eqHigh:-2.5,reverbDecay:2.2,bassPresence:1.8,mudCut:-2},voiceParams:{bass_electric:{cutoff:480}},texture:'vinyl_crackle'}),
ch('synthwave','Synthwave','electronic',{bpm:[105,118],scale:'minor',drums:{pattern:'rock',kick:true,snare:true,hihat:true,clap:true},voices:['supersaw','synth_pad_warm','bass_moog','synth_arp'],leadVoice:'supersaw',density:0.58,reverb:0.42,chord_progression:'i-VII-VI-VII',structure:['intro','verse','chorus','verse','chorus','bridge','outro'],mastering:{eqHigh:1.0,eqMid:0.5,reverbDecay:2.6,bassPresence:1.6},voiceParams:{bass_moog:{reso:4.5}}}),
ch('vaporwave','Vaporwave','electronic',{bpm:[70,85],scale:'major',swing:[0.2,0.3],drums:{pattern:'four_floor',kick:true,clap:true,hihat:true},voices:['synth_pad_warm','wurlitzer','bass_moog','fm_keys','piano_electric'],leadVoice:'wurlitzer',density:0.3,reverb:0.45,filters:{master_lp:9000},chord_progression:'I-vi-IV-V',mastering:{eqLow:3,eqMid:-2,eqHigh:-1,reverbDecay:2.4}}),
ch('deep_house','Deep House','electronic',{bpm:[120,125],scale:'dorian',drums:{pattern:'four_floor',kick:true,clap:true,hihat:true,rimshot:true},voices:['wurlitzer','bass_moog','synth_pad_warm','fm_keys','synth_arp'],leadVoice:'wurlitzer',density:0.5,reverb:0.4,chord_progression:'ii-V-I',mastering:{eqLow:2,bassPresence:1.5}}),
ch('classic_house','Classic House','electronic',{bpm:[125,130],scale:'major',drums:{pattern:'four_floor',kick:true,clap:true,hihat:true},voices:['organ_gospel','synth_brass','bass_electric','piano_electric','synth_pad_warm'],leadVoice:'organ_gospel',density:0.6,reverb:0.3,chord_progression:'I-V-vi-IV',mastering:{eqLow:1,eqHigh:-0.5,bassPresence:1.5}}),
ch('tech_house','Tech House','electronic',{bpm:[127,132],scale:'minor',drums:{pattern:'four_floor',kick:true,clap:true,hihat:true,rimshot:true},voices:['saw_pluck','bass_moog','synth_pad_warm','synth_arp','synth_bell'],leadVoice:'saw_pluck',density:0.45,reverb:0.3,chord_progression:'i-v-VI-iv',mastering:{eqMid:1,bassPresence:1.5}}),
ch('detroit_techno','Detroit Techno','electronic',{bpm:[130,145],scale:'minor',drums:{pattern:'techno',kick:true,hihat:true,clap:true},voices:['saw_pluck','bass_moog','synth_pad_warm','synth_arp','synth_bell'],leadVoice:'saw_pluck',density:0.6,reverb:0.25,chord_progression:'i-VI-III-VII',mastering:{eqMid:1,eqHigh:-0.5,bassPresence:1.5}}),
ch('acid_house','Acid House','electronic',{bpm:[125,135],scale:'minor',drums:{pattern:'four_floor',kick:true,clap:true,hihat:true},voices:['acid_bass','synth_lead_saw'],leadVoice:'acid_bass',density:0.55,reverb:0.3,mastering:{bassPresence:2,mudCut:-2},voiceParams:{acid_bass:{cutoff:300,cutoffPeak:2400}}}),
ch('drum_and_bass','Drum and Bass','electronic',{bpm:[168,175],scale:'minor',drums:{pattern:'dnb',kick:true,snare:true,hihat:true},voices:['reese_bass','synth_pad_warm','supersaw'],leadVoice:'supersaw',density:0.7,reverb:0.35,chord_progression:'i-VI-III-VII',mastering:{eqLow:2,bassPresence:2,mudCut:-2}}),
ch('jungle','Jungle','electronic',{bpm:[160,170],scale:'minor',drums:{pattern:'dnb',kick:true,snare:true,hihat:true},voices:['reese_bass','synth_pad_warm','pizz_strings'],leadVoice:'reese_bass',density:0.65,reverb:0.4,chord_progression:'i-iv-VII-III',mastering:{eqLow:3,bassPresence:2,mudCut:-2}}),
ch('uk_garage','UK Garage','electronic',{bpm:[130,135],scale:'major',swing:[0.5,0.58],drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['wurlitzer','bass_electric','saw_pluck'],leadVoice:'wurlitzer',density:0.6,reverb:0.3,chord_progression:'I-vi-IV-V',mastering:{eqHigh:0.5,bassPresence:1.5}}),
ch('trance','Trance','electronic',{bpm:[138,144],scale:'minor',drums:{pattern:'four_floor',kick:true,clap:true,hihat:true},voices:['supersaw','synth_pad_analog','bass_moog'],leadVoice:'supersaw',density:0.7,reverb:0.4,chord_progression:'vi-IV-I-V',mastering:{eqHigh:0.5,reverbDecay:3,bassPresence:1.5}}),
ch('psytrance','Psytrance','electronic',{bpm:[142,148],scale:'minor',drums:{pattern:'techno',kick:true,hihat:true},voices:['saw_pluck','bass_moog','fm_keys'],leadVoice:'saw_pluck',density:0.7,reverb:0.35,chord_progression:'i-VI-III-VII',mastering:{eqMid:1,bassPresence:2,mudCut:-1.5},voiceParams:{bass_moog:{reso:10}}}),
ch('ambient_techno','Ambient Techno','electronic',{bpm:[120,130],scale:'dorian',drums:{pattern:'four_floor',kick:true,hihat:true},voices:['synth_pad_analog','saw_pluck','bass_moog'],leadVoice:'synth_pad_analog',density:0.3,reverb:0.6,filters:{master_lp:7000},chord_progression:'ii-V-I',mastering:{reverbDecay:4}}),
ch('idm','IDM','electronic',{bpm:[110,150],scale:'whole_tone',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true,rimshot:true},voices:['synth_pluck','fm_keys','synth_bell'],leadVoice:'synth_pluck',density:0.5,reverb:0.4,chord_progression:'I-iii-IV-V',mastering:{eqHigh:0.5}}),
ch('breakbeat','Breakbeat','electronic',{bpm:[128,135],scale:'minor',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['supersaw','bass_electric'],leadVoice:'supersaw',density:0.6,reverb:0.3,chord_progression:'i-VII-VI-V',mastering:{eqLow:1,bassPresence:1.5}}),
ch('future_bass','Future Bass','electronic',{bpm:[140,150],scale:'major',drums:{pattern:'trap',kick:true,snare:true,hihat:true},voices:['synth_pad_glass','bass_wobble','synth_lead_saw'],leadVoice:'synth_pad_glass',density:0.65,reverb:0.4,chord_progression:'I-V-vi-IV',mastering:{eqHigh:0.5}}),
ch('chillwave','Chillwave','electronic',{bpm:[90,110],scale:'major',swing:[0.1,0.2],drums:{pattern:'four_floor',kick:true,clap:true,hihat:true},voices:['synth_pad_analog','wurlitzer','bass_moog'],leadVoice:'synth_pad_analog',density:0.35,reverb:0.65,filters:{master_lp:8500},chord_progression:'I-vi-IV-V',mastering:{reverbDecay:4}}),
ch('electro','Classic Electro','electronic',{bpm:[115,125],scale:'minor',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['bass_808','synth_brass'],leadVoice:'bass_808',density:0.6,reverb:0.25,chord_progression:'i-VII-VI-VII',mastering:{eqLow:2,bassPresence:2,mudCut:-1.5},voiceParams:{bass_808:{harmonics:0.2}}}),
ch('footwork','Footwork','electronic',{bpm:[158,164],scale:'whole_tone',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['synth_bell','bass_808'],leadVoice:'synth_bell',density:0.7,reverb:0.3,chord_progression:'I-iii-IV-V',mastering:{bassPresence:2,mudCut:-1.5},voiceParams:{bass_808:{harmonics:0.2}}}),
ch('jersey_club','Jersey Club','electronic',{bpm:[142,150],scale:'minor',swing:[0.6,0.66],drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['bass_808','saw_pluck'],leadVoice:'bass_808',density:0.65,reverb:0.3,chord_progression:'i-iv-VII-III',mastering:{bassPresence:2,mudCut:-1.5},voiceParams:{bass_808:{harmonics:0.2}}}),
ch('hyperpop','Hyperpop','electronic',{bpm:[150,165],scale:'major',drums:{pattern:'trap',kick:true,snare:true,hihat:true},voices:['synth_lead_square','synth_bell','bass_wobble'],leadVoice:'synth_lead_square',density:0.85,reverb:0.3,chord_progression:'I-V-vi-IV',mastering:{eqHigh:1.5,compRatio:4,bassPresence:1.5}}),
ch('hardstyle','Hardstyle','electronic',{bpm:[150,160],scale:'minor',drums:{pattern:'techno',kick:true,hihat:true},voices:['supersaw','bass_moog'],leadVoice:'supersaw',density:0.8,reverb:0.3,mastering:{eqLow:3,compRatio:4,bassPresence:2}}),
ch('grime','Grime','electronic',{bpm:[139,141],scale:'minor',drums:{pattern:'trap',kick:true,snare:true,hihat:true},voices:['bass_808','synth_lead_square'],leadVoice:'bass_808',density:0.6,reverb:0.25,chord_progression:'i-VI-III-VII',mastering:{eqLow:2,bassPresence:2,mudCut:-1.5},voiceParams:{bass_808:{harmonics:0.18}}}),
ch('dubstep','Classic Dubstep','electronic',{bpm:[138,142],scale:'minor',drums:{pattern:'dubstep',kick:true,snare:true,hihat:true},voices:['reese_bass','synth_pad_warm','bass_wobble'],leadVoice:'reese_bass',density:0.5,reverb:0.35,chord_progression:'i-VI-III-VII',mastering:{eqLow:3,bassPresence:3,mudCut:-2.5}}),

/* ── HIP HOP (10) ── */
ch('boom_bap','Boom Bap','hiphop',{bpm:[86,96],scale:'minor_pentatonic',swing:[0.58,0.64],drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true,rimshot:true},voices:['piano_chop','bass_808','guitar_muted'],leadVoice:'piano_chop',density:0.36,reverb:0.35,filters:{master_lp:9000},chord_progression:'i-VII-VI-VII',structure:['intro','verse','chorus','verse','chorus','outro'],mastering:{eqLow:2.5,eqMid:0.5,eqHigh:-1.5,bassPresence:2,mudCut:-2},voiceParams:{bass_electric:{cutoff:550}}}),
ch('trap','Trap','hiphop',{bpm:[134,148],scale:'minor',drums:{pattern:'trap',kick:true,snare:true,hihat:true},voices:['bass_808','reese_bass','saw_pluck','synth_bell'],leadVoice:'saw_pluck',density:0.28,reverb:0.3,chord_progression:'i-VI-III-VII',structure:['intro','verse','chorus','verse','chorus','bridge','outro'],mastering:{eqLow:3.5,eqHigh:0.0,bassPresence:3,mudCut:-2.5}}),
ch('cloud_rap','Cloud Rap','hiphop',{bpm:[60,80],scale:'minor',drums:{pattern:'lofi',kick:true,snare:true,hihat:true},voices:['synth_pad_analog','piano_chop','bass_electric'],leadVoice:'synth_pad_analog',density:0.3,reverb:0.7,filters:{master_lp:7500},chord_progression:'i-VI-III-VII',mastering:{reverbDecay:4}}),
ch('phonk','Phonk','hiphop',{bpm:[130,145],scale:'minor',drums:{pattern:'trap',kick:true,snare:true,hihat:true,cowbell:true},voices:['bass_808','cowbell'],leadVoice:'cowbell',density:0.55,reverb:0.3,filters:{master_lp:8000},chord_progression:'i-VII-VI-V',mastering:{eqLow:2,bassPresence:2,mudCut:-2},voiceParams:{bass_808:{harmonics:0.2}}}),
ch('g_funk','G-Funk','hiphop',{bpm:[90,105],scale:'minor_pentatonic',swing:[0.6,0.66],drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true},voices:['synth_brass','bass_moog','piano_chop'],leadVoice:'synth_brass',density:0.45,reverb:0.3,chord_progression:'i-VII-VI-VII',mastering:{bassPresence:1.5},leadVoiceParams:{cutoff:2500}}),
ch('chicago_drill','Chicago Drill','hiphop',{bpm:[60,75],scale:'minor',drums:{pattern:'trap',kick:true,snare:true,hihat:true},voices:['bass_808','synth_pad_warm','synth_brass'],leadVoice:'bass_808',density:0.5,reverb:0.3,chord_progression:'i-VI-III-VII',mastering:{eqLow:3,bassPresence:2,mudCut:-1.5},voiceParams:{bass_808:{harmonics:0.15}}}),
ch('uk_drill','UK Drill','hiphop',{bpm:[130,140],scale:'minor',swing:[0.5,0.58],drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['bass_808','saw_pluck','synth_pad_warm'],leadVoice:'saw_pluck',density:0.55,reverb:0.3,chord_progression:'i-VI-III-VII',mastering:{eqLow:2,bassPresence:2,mudCut:-1.5},voiceParams:{bass_808:{harmonics:0.15}}}),
ch('neo_soul','Neo-Soul','hiphop',{bpm:[85,100],scale:'dorian',swing:[0.55,0.6],drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true,rimshot:true},voices:['wurlitzer','bass_electric','guitar_muted'],leadVoice:'wurlitzer',density:0.5,reverb:0.35,chord_progression:'ii-V-I',mastering:{eqLow:1,eqHigh:-0.5,bassPresence:1}}),
ch('classic_rnb','Classic R&B','hiphop',{bpm:[90,110],scale:'major',swing:[0.5,0.56],drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true},voices:['piano_electric','bass_electric','harmonica'],leadVoice:'piano_electric',density:0.45,reverb:0.4,chord_progression:'I-V-vi-IV',mastering:{eqLow:1,bassPresence:1}}),
ch('contemporary_rnb','Contemporary R&B','hiphop',{bpm:[85,100],scale:'minor',drums:{pattern:'trap',kick:true,snare:true,hihat:true},voices:['wurlitzer','piano_electric','bass_electric'],leadVoice:'wurlitzer',density:0.35,reverb:0.45,chord_progression:'i-VI-III-VII',mastering:{reverbDecay:3,bassPresence:1}}),

/* ── JAZZ (10) ── */
ch('bebop','Bebop','jazz',{bpm:[150,180],scale:'major',swing:[0.62,0.66],drums:{pattern:'boom_bap',kick:false,snare:true,hihat:true,ride:true},voices:['piano_acoustic','bass_acoustic'],leadVoice:'piano_acoustic',density:0.7,reverb:0.25,chord_progression:'ii-V-I-VI',mastering:{eqMid:1}}),
ch('cool_jazz','Cool Jazz','jazz',{bpm:[90,110],scale:'dorian',swing:[0.6,0.64],drums:{pattern:'bossa',kick:true,hihat:true,ride:true},voices:['sax','piano_acoustic','bass_acoustic'],leadVoice:'sax',density:0.4,reverb:0.45,chord_progression:'ii-V-I',mastering:{eqLow:1}}),
ch('modal_jazz','Modal Jazz','jazz',{bpm:[100,120],scale:'dorian',swing:[0.6,0.65],drums:{pattern:'bossa',kick:true,hihat:true,ride:true},voices:['sax','piano_acoustic','bass_acoustic'],leadVoice:'sax',density:0.5,reverb:0.4,chord_progression:'i-VI-III-VII',mastering:{eqMid:1}}),
ch('jazz_fusion','Jazz Fusion','jazz',{bpm:[120,150],scale:'mixolydian',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true,ride:true},voices:['wurlitzer','bass_electric','synth_brass'],leadVoice:'wurlitzer',density:0.65,reverb:0.3,chord_progression:'ii-V-I',mastering:{eqMid:2,bassPresence:1}}),
ch('nu_jazz','Nu Jazz','jazz',{bpm:[110,130],scale:'dorian',swing:[0.55,0.6],drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['fm_keys','bass_moog','flute_breathy'],leadVoice:'fm_keys',density:0.5,reverb:0.4,chord_progression:'ii-V-I',mastering:{eqHigh:-0.5}}),
ch('smooth_jazz','Smooth Jazz','jazz',{bpm:[90,110],scale:'major',swing:[0.55,0.6],drums:{pattern:'bossa',kick:true,hihat:true,ride:true},voices:['sax','piano_electric','bass_electric'],leadVoice:'sax',density:0.4,reverb:0.5,chord_progression:'I-vi-IV-V',mastering:{eqLow:1,reverbDecay:3,bassPresence:1}}),
ch('bossa_nova','Bossa Nova','jazz',{bpm:[112,128],scale:'major',swing:[0.54,0.58],drums:{pattern:'bossa',kick:true,hihat:true,rimshot:true},voices:['guitar_clean','bass_acoustic','pan_flute','vibraphone'],leadVoice:'guitar_clean',density:0.42,reverb:0.38,chord_progression:'ii-V-I',structure:['intro','verse','chorus','verse','chorus','outro'],mastering:{eqLow:1,eqHigh:-1.0,reverbDecay:2.4}}),
ch('samba','Samba','jazz',{bpm:[140,170],scale:'major',drums:{pattern:'samba',kick:true,snare:true,hihat:true,shaker:true},voices:['accordion','pizz_strings','bass_acoustic'],leadVoice:'accordion',density:0.6,reverb:0.3,chord_progression:'I-IV-V-I',mastering:{eqMid:1}}),
ch('big_band','Big Band','jazz',{bpm:[130,170],scale:'major',swing:[0.62,0.67],drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true,ride:true},voices:['synth_brass','sax','bass_acoustic'],leadVoice:'synth_brass',density:0.65,reverb:0.3,chord_progression:'I-vi-IV-V',mastering:{eqMid:1,eqHigh:-0.5}}),
ch('free_jazz','Free Jazz','jazz',{bpm:[120,200],scale:'chromatic',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true,ride:true},voices:['sax','piano_acoustic','bass_acoustic'],leadVoice:'sax',density:0.7,reverb:0.3,chord_progression:null,mastering:{eqMid:1}}),

/* ── CLASSICAL (10) ── */
ch('baroque','Baroque','classical',{bpm:[80,110],scale:'major',beatless:false,drums:{pattern:'silence'},voices:['harpsichord','pizz_strings','strings_section'],leadVoice:'harpsichord',density:0.6,reverb:0.55,chord_progression:'I-IV-V-I',mastering:{reverbDecay:3.5}}),
ch('romantic_piano','Romantic Piano','classical',{bpm:[60,90],scale:'minor',beatless:false,drums:{pattern:'silence'},voices:['piano_acoustic','strings_section','bass_acoustic','flute_breathy'],leadVoice:'piano_acoustic',density:0.5,reverb:0.6,chord_progression:'i-VI-III-VII',tempo_humanise:0.06,mastering:{reverbDecay:4}}),
ch('minimalist','Minimalist','classical',{bpm:[70,90],scale:'major',drums:{pattern:'silence'},voices:['piano_acoustic','vibraphone','music_box'],leadVoice:'piano_acoustic',density:0.4,reverb:0.5,chord_progression:'I-V-vi-IV',mastering:{reverbDecay:3}}),
ch('orchestral','Orchestral Cinematic','classical',{bpm:[70,110],scale:'minor',drums:{pattern:'silence'},voices:['strings_section','brass_section','choir_aaah','pizz_strings'],leadVoice:'strings_section',density:0.5,reverb:0.6,chord_progression:'i-VI-III-VII',mastering:{reverbDecay:4}}),
ch('string_quartet','String Quartet','classical',{bpm:[80,110],scale:'major',drums:{pattern:'silence'},voices:['strings_section','pizz_strings'],leadVoice:'strings_section',density:0.5,reverb:0.45,chord_progression:'I-IV-V-I',mastering:{reverbDecay:3}}),
ch('piano_solo','Piano Solo','classical',{bpm:[70,110],scale:'major',drums:{pattern:'silence'},voices:['piano_acoustic','strings_section','bass_acoustic','flute_breathy'],leadVoice:'piano_acoustic',density:0.5,reverb:0.55,chord_progression:'I-vi-IV-V',tempo_humanise:0.05,mastering:{reverbDecay:3.5}}),
ch('choral','Choral','classical',{bpm:[60,80],scale:'major',drums:{pattern:'silence'},voices:['choir_aaah','strings_section','ocarina','flute_breathy','piano_acoustic'],leadVoice:'choir_aaah',density:0.3,reverb:0.7,beatless:true,chord_progression:'I-IV-V-I',mastering:{reverbDecay:5}}),
ch('impressionist','Impressionist','classical',{bpm:[60,90],scale:'whole_tone',drums:{pattern:'silence'},voices:['piano_acoustic','celesta','music_box','strings_section','flute_breathy','pizz_strings'],leadVoice:'piano_acoustic',density:0.4,reverb:0.65,beatless:true,chord_progression:'I-iii-IV-V',tempo_humanise:0.07,mastering:{reverbDecay:4}}),
ch('contemporary_classical','Contemporary Classical','classical',{bpm:[70,110],scale:'diminished',drums:{pattern:'silence'},voices:['piano_acoustic','pizz_strings','strings_section'],leadVoice:'piano_acoustic',density:0.5,reverb:0.5,chord_progression:'i-iv-VII-III',mastering:{reverbDecay:3}}),
ch('neo_romantic','Neo-Romantic','classical',{bpm:[70,100],scale:'minor',drums:{pattern:'silence'},voices:['strings_section','piano_acoustic','brass_section'],leadVoice:'strings_section',density:0.5,reverb:0.6,chord_progression:'i-VI-III-VII',mastering:{reverbDecay:4}}),

/* ── WORLD (15) ── */
ch('reggae','Reggae','world',{bpm:[70,85],scale:'major',swing:[0.5,0.56],drums:{pattern:'reggae',kick:true,snare:true,hihat:true},voices:['guitar_muted','bass_electric','organ_hammond'],leadVoice:'guitar_muted',density:0.45,reverb:0.4,chord_progression:'I-IV-V-I',mastering:{eqLow:2,bassPresence:2}}),
ch('dub','Dub','world',{bpm:[70,85],scale:'minor',drums:{pattern:'reggae',kick:true,snare:true,hihat:true},voices:['bass_electric','synth_pad_warm','organ_hammond'],leadVoice:'bass_electric',density:0.35,reverb:0.7,chord_progression:'i-iv-VII-III',mastering:{eqLow:3,reverbDecay:5,bassPresence:3,mudCut:-2}}),
ch('afrobeat','Afrobeat','world',{bpm:[110,125],scale:'dorian',drums:{pattern:'afrobeat',kick:true,snare:true,hihat:true,shaker:true},voices:['sax','piano_acoustic','bass_electric'],leadVoice:'sax',density:0.6,reverb:0.35,chord_progression:'ii-V-I',mastering:{eqMid:1}}),
ch('cumbia','Cumbia','world',{bpm:[90,110],scale:'major',drums:{pattern:'samba',kick:true,hihat:true,shaker:true},voices:['accordion','bass_electric','flute_breathy'],leadVoice:'accordion',density:0.5,reverb:0.35,chord_progression:'I-IV-V-I',mastering:{eqLow:1}}),
ch('salsa','Salsa','world',{bpm:[150,180],scale:'major',drums:{pattern:'samba',kick:true,snare:true,hihat:true,shaker:true,clave:true},voices:['piano_acoustic','bass_acoustic','synth_brass'],leadVoice:'piano_acoustic',density:0.65,reverb:0.3,chord_progression:'I-IV-V-I',mastering:{eqMid:1}}),
ch('flamenco','Flamenco','world',{bpm:[100,130],scale:'phrygian',drums:{pattern:'samba',kick:true,snare:true,hihat:true,clap:true},voices:['guitar_clean','bass_acoustic','accordion'],leadVoice:'guitar_clean',density:0.6,reverb:0.4,chord_progression:'i-VII-VI-V',mastering:{eqMid:1}}),
ch('celtic_folk','Celtic Folk','world',{bpm:[110,150],scale:'dorian',drums:{pattern:'samba',kick:true,hihat:true,shaker:true},voices:['pan_flute','dulcimer','bass_acoustic'],leadVoice:'pan_flute',density:0.55,reverb:0.4,chord_progression:'i-VII-VI-VII',mastering:{eqHigh:-0.5}}),
ch('maqam','Middle Eastern Maqam','world',{bpm:[90,120],scale:'maqam_hijaz',drums:{pattern:'afrobeat',kick:true,hihat:true,perc:true},voices:['sitar_drone','ocarina','bass_acoustic'],leadVoice:'ocarina',density:0.5,reverb:0.5,chord_progression:'i-VII-VI-V',mastering:{reverbDecay:3.5}}),
ch('raga','Indian Raga','world',{bpm:[70,100],scale:'raga_bhairav',drums:{pattern:'silence'},voices:['sitar_drone','pan_flute'],leadVoice:'pan_flute',density:0.4,reverb:0.6,beatless:true,chord_progression:'i-iv-VII-III',mastering:{reverbDecay:4}}),
ch('kankyo','Japanese Ambient','world',{bpm:[50,70],scale:'japanese_hirajoshi',drums:{pattern:'silence'},voices:['koto','vibraphone','synth_pad_warm'],leadVoice:'koto',density:0.2,reverb:0.8,beatless:true,chord_progression:'I-iii-IV-V',mastering:{reverbDecay:5}}),
ch('gamelan','Gamelan','world',{bpm:[80,110],scale:'gamelan_pelog',drums:{pattern:'silence'},voices:['vibraphone','music_box','kalimba'],leadVoice:'vibraphone',density:0.5,reverb:0.5,chord_progression:'I-iii-IV-V',mastering:{reverbDecay:3}}),
ch('african_polyrhythm','African Polyrhythm','world',{bpm:[110,130],scale:'major_pentatonic',drums:{pattern:'afrobeat',kick:true,snare:true,hihat:true,shaker:true,djembe:true},voices:['kalimba','marimba','bass_acoustic'],leadVoice:'kalimba',density:0.6,reverb:0.35,chord_progression:'I-IV-V-I',mastering:{eqMid:1}}),
ch('steel_pan','Caribbean Steel Pan','world',{bpm:[100,130],scale:'major_pentatonic',drums:{pattern:'samba',kick:true,hihat:true,shaker:true},voices:['music_box','vibraphone','bass_electric'],leadVoice:'music_box',density:0.55,reverb:0.4,chord_progression:'I-V-vi-IV',mastering:{eqHigh:-0.5}}),
ch('fado','Fado','world',{bpm:[70,90],scale:'phrygian',drums:{pattern:'silence'},voices:['guitar_clean','accordion','choir_aaah'],leadVoice:'guitar_clean',density:0.4,reverb:0.55,chord_progression:'i-VII-VI-V',mastering:{reverbDecay:3.5}}),
ch('klezmer','Klezmer','world',{bpm:[100,130],scale:'harmonic_minor',drums:{pattern:'samba',kick:true,snare:true,hihat:true,clarinet:true},voices:['accordion','flute_breathy','bass_acoustic'],leadVoice:'accordion',density:0.6,reverb:0.35,chord_progression:'i-iv-VII-III',mastering:{eqMid:1}}),

/* ── ROCK / METAL (10) ── */
ch('classic_rock','Classic Rock','rock',{bpm:[110,130],scale:'major_pentatonic',drums:{pattern:'rock',kick:true,snare:true,hihat:true,crash:true},voices:['guitar_distorted','bass_electric','organ_hammond'],leadVoice:'guitar_distorted',density:0.7,reverb:0.25,chord_progression:'I-IV-V-I',mastering:{eqLow:2,eqHigh:-0.5}}),
ch('heavy_metal','Heavy Metal','rock',{bpm:[140,170],scale:'minor',drums:{pattern:'metal',kick:true,snare:true,hihat:true,crash:true},voices:['guitar_distorted','bass_electric','organ_hammond','synth_brass','piano_electric'],leadVoice:'guitar_distorted',density:0.8,reverb:0.2,chord_progression:'i-VI-III-VII',mastering:{eqLow:2,eqHigh:0.5,compRatio:4}}),
ch('doom_metal','Doom Metal','rock',{bpm:[50,70],scale:'minor',drums:{pattern:'rock',kick:true,snare:true,hihat:true,crash:true},voices:['guitar_distorted','reese_bass'],leadVoice:'guitar_distorted',density:0.6,reverb:0.3,chord_progression:'i-bVI-bVII-i',mastering:{eqLow:4,compRatio:4,bassPresence:3,mudCut:-2}}),
ch('post_rock','Post-Rock','rock',{bpm:[90,120],scale:'minor',drums:{pattern:'rock',kick:true,snare:true,hihat:true,crash:true},voices:['guitar_clean','guitar_distorted','bass_electric','synth_pad_analog'],leadVoice:'guitar_clean',density:0.5,reverb:0.5,chord_progression:'i-VI-III-VII',mastering:{reverbDecay:4,bassPresence:1}}),
ch('shoegaze','Shoegaze','rock',{bpm:[100,130],scale:'major',drums:{pattern:'rock',kick:true,snare:true,hihat:true,crash:true},voices:['guitar_distorted','supersaw','bass_electric'],leadVoice:'guitar_distorted',density:0.6,reverb:0.7,chord_progression:'I-vi-IV-V',mastering:{reverbDecay:5,eqHigh:0.5}}),
ch('grunge','Grunge','rock',{bpm:[100,130],scale:'minor',drums:{pattern:'rock',kick:true,snare:true,hihat:true,crash:true},voices:['guitar_distorted','bass_electric','organ_hammond','synth_brass','piano_electric'],leadVoice:'guitar_distorted',density:0.7,reverb:0.25,chord_progression:'i-VI-III-VII',mastering:{eqLow:2}}),
ch('punk','Punk','rock',{bpm:[150,180],scale:'major',drums:{pattern:'metal',kick:true,snare:true,hihat:true,crash:true},voices:['guitar_distorted','bass_electric','organ_hammond','synth_brass','piano_electric'],leadVoice:'guitar_distorted',density:0.85,reverb:0.15,chord_progression:'I-IV-V-I',mastering:{eqHigh:0.5}}),
ch('math_rock','Math Rock','rock',{bpm:[110,140],scale:'major',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true,rimshot:true},voices:['guitar_clean','bass_electric','fm_keys'],leadVoice:'guitar_clean',density:0.65,reverb:0.3,chord_progression:'I-iii-IV-V',mastering:{eqMid:1}}),
ch('prog_rock','Prog Rock','rock',{bpm:[100,140],scale:'minor',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true,ride:true},voices:['synth_brass','guitar_distorted','bass_moog','fm_keys'],leadVoice:'synth_brass',density:0.7,reverb:0.35,chord_progression:'i-VI-III-VII',mastering:{eqMid:1,bassPresence:1}}),
ch('black_metal_atmo','Atmospheric Black Metal','rock',{bpm:[150,180],scale:'minor',drums:{pattern:'metal',kick:true,snare:true,hihat:true,crash:true},voices:['guitar_distorted','synth_pad_analog'],leadVoice:'guitar_distorted',density:0.85,reverb:0.6,chord_progression:'i-VI-III-VII',mastering:{eqHigh:1.5,reverbDecay:4}}),

/* ── AMBIENT / EXPERIMENTAL (15) ── */
ch('dark_ambient','Dark Ambient','ambient',{bpm:[40,60],scale:'minor',drums:{pattern:'silence'},voices:['synth_pad_analog','ambient_pad'],leadVoice:'synth_pad_analog',density:0.1,reverb:0.9,beatless:true,chord_progression:'i-bVI-bVII-i',mastering:{reverbDecay:6}}),
ch('drone','Drone','ambient',{bpm:[30,45],scale:'major',drums:{pattern:'silence'},voices:['ambient_pad','synth_pad_analog'],leadVoice:'ambient_pad',density:0.05,reverb:0.95,beatless:true,chord_progression:'I-IV-I',mastering:{reverbDecay:7}}),
ch('noise_structured','Structured Noise','ambient',{bpm:[80,120],scale:'chromatic',drums:{pattern:'silence'},voices:['synth_brass','guitar_distorted'],leadVoice:'synth_brass',density:0.5,reverb:0.5,beatless:true,chord_progression:null,mastering:{compRatio:6}}),
ch('glitch','Glitch','ambient',{bpm:[120,160],scale:'whole_tone',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true,rimshot:true},voices:['synth_pluck','fm_keys','music_box'],leadVoice:'synth_pluck',density:0.6,reverb:0.3,chord_progression:'I-iii-IV-V',mastering:{eqHigh:0.5}}),
ch('microsound','Microsound','ambient',{bpm:[40,60],scale:'major_pentatonic',drums:{pattern:'silence'},voices:['music_box','ambient_pad'],leadVoice:'music_box',density:0.08,reverb:0.85,beatless:true,chord_progression:'I-V-vi-IV',mastering:{reverbDecay:6}}),
ch('musique_concrete','Musique Concrète','ambient',{bpm:[60,90],scale:'chromatic',drums:{pattern:'silence'},voices:['synth_pad_glass','ambient_pad','theremin','synth_pad_warm'],leadVoice:'synth_pad_glass',density:0.3,reverb:0.5,beatless:true,chord_progression:null,mastering:{reverbDecay:4}}),
ch('new_age','New Age','ambient',{bpm:[60,80],scale:'major_pentatonic',drums:{pattern:'silence'},voices:['synth_pad_warm','ocarina','kalimba','synth_pad_glass','music_box','theremin'],leadVoice:'synth_pad_warm',density:0.2,reverb:0.7,beatless:true,chord_progression:'I-V-vi-IV',mastering:{reverbDecay:5}}),
ch('binaural_healing','Binaural / Healing','ambient',{bpm:[40,55],scale:'major',drums:{pattern:'silence'},voices:['ambient_pad','synth_pad_analog','ocarina','synth_pad_glass','kalimba','music_box'],leadVoice:'ambient_pad',density:0.08,reverb:0.9,beatless:true,special:'binaural',chord_progression:'I-IV-V-I',mastering:{reverbDecay:7}}),
ch('lowercase','Lowercase','ambient',{bpm:[30,45],scale:'major_pentatonic',drums:{pattern:'silence'},voices:['kalimba','ambient_pad','music_box','synth_pad_glass','theremin'],leadVoice:'kalimba',density:0.04,reverb:0.9,beatless:true,chord_progression:'I-V-vi-IV',mastering:{reverbDecay:7}}),
ch('harsh_noise','Harsh Noise','ambient',{bpm:[80,140],scale:'chromatic',drums:{pattern:'silence'},voices:['guitar_distorted','synth_brass'],leadVoice:'guitar_distorted',density:0.7,reverb:0.4,beatless:true,chord_progression:null,mastering:{compRatio:8,eqHigh:1.5}}),
ch('spectral','Spectral Music','ambient',{bpm:[60,90],scale:'whole_tone',drums:{pattern:'silence'},voices:['synth_pad_glass','synth_bell','celesta'],leadVoice:'synth_pad_glass',density:0.3,reverb:0.7,beatless:true,chord_progression:'I-iii-IV-V',mastering:{reverbDecay:5}}),
ch('post_industrial','Post-Industrial','ambient',{bpm:[110,130],scale:'minor',drums:{pattern:'techno',kick:true,hihat:true},voices:['saw_pluck','reese_bass'],leadVoice:'saw_pluck',density:0.6,reverb:0.4,chord_progression:'i-VI-III-VII',mastering:{eqMid:2,compRatio:4,bassPresence:2,mudCut:-1.5}}),
ch('witch_house','Witch House','ambient',{bpm:[50,70],scale:'minor',drums:{pattern:'dubstep',kick:true,snare:true,hihat:true},voices:['synth_pad_warm','reese_bass','synth_bell'],leadVoice:'synth_pad_warm',density:0.3,reverb:0.8,chord_progression:'i-bVI-bVII-i',mastering:{reverbDecay:6,bassPresence:2,mudCut:-1.5}}),
ch('darkwave','Darkwave','ambient',{bpm:[100,130],scale:'minor',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['synth_brass','synth_pad_analog','bass_moog'],leadVoice:'synth_brass',density:0.55,reverb:0.5,chord_progression:'i-VI-III-VII',mastering:{reverbDecay:4,bassPresence:1.5}}),
ch('deconstructed_club','Deconstructed Club','ambient',{bpm:[120,150],scale:'chromatic',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['bass_wobble','synth_pad_glass'],leadVoice:'bass_wobble',density:0.5,reverb:0.5,chord_progression:null,mastering:{eqHigh:0.5}}),
ch('dungeon_synth','Dungeon Synth','ambient',{bpm:[40,60],scale:'minor',drums:{pattern:'silence'},voices:['synth_pad_warm','organ_pipe','music_box','choir_aaah'],leadVoice:'synth_pad_warm',density:0.12,reverb:0.85,beatless:true,chord_progression:'i-bVI-bVII-i',mastering:{reverbDecay:6}}),

/* ── RETRO / CHIPTUNE (10) ── */
ch('chiptune_nes','Chiptune (NES)','retro',{bpm:[120,150],scale:'major_pentatonic',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['chiptune_pulse','chiptune_triangle'],leadVoice:'chiptune_pulse',density:0.75,reverb:0.02,chord_progression:'I-V-vi-IV',mastering:{eqHigh:0.5,eqMid:1,reverbDecay:0.4}}),
ch('chiptune_sid','Chiptune (C64 SID)','retro',{bpm:[120,150],scale:'minor',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['chiptune_saw','chiptune_pulse'],leadVoice:'chiptune_saw',density:0.7,reverb:0.12,chord_progression:'i-VII-VI-VII',mastering:{eqHigh:-0.5}}),
ch('chiptune_gb','Chiptune (Game Boy)','retro',{bpm:[120,160],scale:'major',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['chiptune_pulse','chiptune_triangle'],leadVoice:'chiptune_pulse',density:0.7,reverb:0.1,chord_progression:'I-IV-V-I',mastering:{eqHigh:-0.5}}),
ch('arcade','Arcade Cabinet','retro',{bpm:[130,160],scale:'major_pentatonic',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['ym2612_lead','ym2612_bass','chiptune_pulse'],leadVoice:'ym2612_lead',density:0.75,reverb:0.08,chord_progression:'I-V-vi-IV',mastering:{eqHigh:-0.5}}),
ch('demoscene','Demoscene / Tracker','retro',{bpm:[125,160],scale:'minor',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['ym2612_lead','chiptune_saw','ym2612_bass'],leadVoice:'ym2612_lead',density:0.7,reverb:0.1,chord_progression:'i-VII-VI-V',mastering:{eqHigh:-0.5}}),
ch('elevator','Elevator Music','retro',{bpm:[90,110],scale:'major',drums:{pattern:'four_floor',kick:true,hihat:true},voices:['wurlitzer','strings_section','bass_electric'],leadVoice:'wurlitzer',density:0.4,reverb:0.4,chord_progression:'I-vi-IV-V',mastering:{eqLow:1,bassPresence:1}}),
ch('lounge','Lounge / Exotica','retro',{bpm:[100,120],scale:'major',swing:[0.55,0.58],drums:{pattern:'bossa',kick:true,hihat:true,rimshot:true},voices:['vibraphone','bass_acoustic','pan_flute'],leadVoice:'vibraphone',density:0.5,reverb:0.4,chord_progression:'ii-V-I',mastering:{eqLow:1}}),
ch('disco','Disco','retro',{bpm:[118,124],scale:'major',drums:{pattern:'four_floor',kick:true,clap:true,hihat:true},voices:['synth_brass','strings_section','bass_electric','organ_hammond'],leadVoice:'synth_brass',density:0.68,reverb:0.32,chord_progression:'I-vi-IV-V',structure:['intro','verse','chorus','verse','chorus','bridge','outro'],mastering:{eqHigh:1.0,eqMid:1,bassPresence:1.8}}),
ch('funk','Funk','retro',{bpm:[100,115],scale:'dorian',swing:[0.6,0.64],drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['guitar_muted','bass_electric','clavinet','harmonica'],leadVoice:'guitar_muted',density:0.7,reverb:0.25,chord_progression:'i-iv-VII-III',mastering:{eqMid:1,bassPresence:1.5}}),
ch('city_pop','City Pop','retro',{bpm:[98,112],scale:'major',swing:[0.52,0.58],drums:{pattern:'city_pop',kick:true,snare:true,hihat:true,clap:true},voices:['fm_keys','bass_electric','synth_brass','guitar_clean'],leadVoice:'fm_keys',density:0.52,reverb:0.38,chord_progression:'IV-V-I-vi',structure:['intro','verse','chorus','verse','chorus','bridge','outro'],mastering:{eqLow:1.5,eqHigh:0.0,reverbDecay:2.8,bassPresence:1.4}}),

/* ── FOLK / COUNTRY (8) ── */
ch('indie_folk','Indie Folk','folk',{bpm:[80,100],scale:'major',drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true},voices:['guitar_clean','banjo','bass_acoustic','harmonica','dulcimer','flute_breathy'],leadVoice:'guitar_clean',density:0.35,reverb:0.4,chord_progression:'I-V-vi-IV',mastering:{eqLow:1}}),
ch('appalachian','Appalachian Folk','folk',{bpm:[100,130],scale:'mixolydian',drums:{pattern:'samba',kick:true,hihat:true,shaker:true},voices:['dulcimer','banjo','bass_acoustic'],leadVoice:'dulcimer',density:0.5,reverb:0.35,chord_progression:'I-IV-V-I',mastering:{eqHigh:-0.5}}),
ch('bluegrass','Bluegrass','folk',{bpm:[120,160],scale:'major_pentatonic',drums:{pattern:'samba',kick:true,hihat:true,shaker:true},voices:['banjo','guitar_clean','bass_acoustic'],leadVoice:'banjo',density:0.65,reverb:0.3,chord_progression:'I-IV-V-I',mastering:{eqMid:1}}),
ch('classic_country','Classic Country','folk',{bpm:[90,110],scale:'major',drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true},voices:['guitar_clean','bass_electric','harmonica'],leadVoice:'guitar_clean',density:0.45,reverb:0.3,chord_progression:'I-IV-V-I',mastering:{eqHigh:-0.5,bassPresence:1}}),
ch('gospel','Gospel','folk',{bpm:[70,90],scale:'major',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true,clap:true},voices:['organ_gospel','choir_aaah','piano_acoustic','harmonica'],leadVoice:'organ_gospel',density:0.5,reverb:0.5,chord_progression:'I-IV-V-I',mastering:{reverbDecay:3.5}}),
ch('blues','Blues','folk',{bpm:[70,90],scale:'blues',swing:[0.62,0.66],drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true},voices:['harmonica','guitar_clean','bass_electric'],leadVoice:'harmonica',density:0.5,reverb:0.3,chord_progression:'I-IV-V-I',mastering:{eqLow:1,bassPresence:1}}),
ch('delta_blues','Delta Blues','folk',{bpm:[60,80],scale:'blues',drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true},voices:['guitar_clean','bass_acoustic','harmonica'],leadVoice:'guitar_clean',density:0.45,reverb:0.35,chord_progression:'I-IV-V-I',mastering:{eqLow:2}}),
ch('honky_tonk','Honky Tonk','folk',{bpm:[90,110],scale:'major',swing:[0.5,0.55],drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true},voices:['piano_acoustic','bass_electric','accordion'],leadVoice:'piano_acoustic',density:0.5,reverb:0.3,chord_progression:'I-IV-V-I',mastering:{eqMid:1}}),

/* ── POP (7) ── */
ch('baroque_pop','Baroque Pop','pop',{bpm:[90,110],scale:'major',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['harpsichord','pizz_strings','bass_electric'],leadVoice:'harpsichord',density:0.5,reverb:0.45,chord_progression:'I-vi-IV-V',mastering:{reverbDecay:3}}),
ch('chamber_pop','Chamber Pop','pop',{bpm:[90,110],scale:'major',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['piano_acoustic','pizz_strings','celesta'],leadVoice:'piano_acoustic',density:0.5,reverb:0.45,chord_progression:'I-vi-IV-V',mastering:{reverbDecay:3}}),
ch('dream_pop','Dream Pop','pop',{bpm:[90,110],scale:'major',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['guitar_distorted','synth_pad_analog','bass_electric'],leadVoice:'guitar_distorted',density:0.45,reverb:0.7,chord_progression:'I-vi-IV-V',mastering:{reverbDecay:5}}),
ch('indie_pop','Indie Pop','pop',{bpm:[112,128],scale:'major',drums:{pattern:'jpop',kick:true,snare:true,hihat:true},voices:['guitar_clean','fm_keys','bass_electric','synth_pad_warm'],leadVoice:'guitar_clean',density:0.55,reverb:0.36,chord_progression:'I-V-vi-IV',structure:['intro','verse','chorus','verse','chorus','bridge','outro'],mastering:{eqHigh:0.0,eqMid:0.5}}),
ch('bedroom_pop','Bedroom Pop','pop',{bpm:[80,100],scale:'major',drums:{pattern:'lofi',kick:true,snare:true,hihat:true},voices:['wurlitzer','guitar_clean','bass_electric'],leadVoice:'wurlitzer',density:0.4,reverb:0.5,filters:{master_lp:8500},chord_progression:'I-vi-IV-V',mastering:{reverbDecay:3.5,bassPresence:1}}),
ch('art_pop','Art Pop','pop',{bpm:[90,120],scale:'major',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['fm_keys','synth_pad_glass','synth_bell'],leadVoice:'fm_keys',density:0.55,reverb:0.45,chord_progression:'I-iii-IV-V',mastering:{eqMid:1}}),
ch('synthpop','Synthpop','pop',{bpm:[118,128],scale:'major',drums:{pattern:'synthpop',kick:true,snare:true,hihat:true,clap:true},voices:['supersaw','fm_keys','bass_moog','synth_pad_warm'],leadVoice:'supersaw',density:0.62,reverb:0.34,chord_progression:'I-V-vi-IV',structure:['intro','verse','chorus','verse','chorus','bridge','outro'],mastering:{eqHigh:0.5,eqMid:0.5,bassPresence:1.4}}),

/* ── SPECIAL / WILDCARD (10) ── */

/* ── JAPANESE (10) ── */
ch('jpop','J-Pop','japanese',{bpm:[128,145],scale:'major',drums:{pattern:'jpop',kick:true,snare:true,hihat:true,clap:true},voices:['supersaw','fm_keys','bass_moog','synth_bell'],leadVoice:'supersaw',density:0.7,reverb:0.32,chord_progression:'IV-V-iii-vi',structure:['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],sectionBars:{intro:4,verse:8,prechorus:4,chorus:8,bridge:4,outro:4},mastering:{eqHigh:1.0,eqMid:1,bassPresence:1.3},filters:{}}),
ch('anime_op','Anime OP','japanese',{bpm:[160,185],scale:'major',drums:{pattern:'synthpop',kick:true,snare:true,hihat:true,clap:true,crash:true},voices:['supersaw','ym2612_lead','bass_moog','synth_brass'],leadVoice:'supersaw',density:0.78,reverb:0.28,chord_progression:'IV-V-iii-vi',structure:['intro','verse','prechorus','chorus','verse','chorus','bridge','chorus','outro'],sectionBars:{intro:4,verse:8,prechorus:4,chorus:8,bridge:4,outro:4},mastering:{eqHigh:1.5,eqMid:1.5,bassPresence:1.5}}),
ch('anime_ed','Anime ED','japanese',{bpm:[85,105],scale:'major',drums:{pattern:'city_pop',kick:true,snare:true,hihat:true},voices:['fm_keys','guitar_clean','bass_electric','synth_pad_warm'],leadVoice:'fm_keys',density:0.45,reverb:0.45,chord_progression:'I-vi-IV-V',structure:['intro','verse','chorus','verse','chorus','outro'],mastering:{eqHigh:-0.5,reverbDecay:3,bassPresence:1.2}}),
ch('city_pop_jp','City Pop JP','japanese',{bpm:[100,114],scale:'major',swing:[0.52,0.58],drums:{pattern:'city_pop',kick:true,snare:true,hihat:true,clap:true},voices:['fm_keys','bass_electric','synth_brass','guitar_clean','wurlitzer'],leadVoice:'fm_keys',density:0.55,reverb:0.4,chord_progression:'IV-V-I-vi',structure:['intro','verse','chorus','verse','chorus','bridge','outro'],mastering:{eqLow:1.5,eqHigh:0.30000000000000004,reverbDecay:2.8,bassPresence:1.5}}),
ch('shibuya_kei','Shibuya-kei','japanese',{bpm:[105,125],scale:'major',swing:[0.54,0.6],drums:{pattern:'shibuya',kick:true,snare:true,hihat:true,shaker:true},voices:['vibraphone','bass_acoustic','flute_breathy','wurlitzer'],leadVoice:'vibraphone',density:0.48,reverb:0.4,chord_progression:'ii-V-I',structure:['intro','verse','chorus','verse','chorus','outro'],mastering:{eqHigh:0.0,reverbDecay:2.6}}),
ch('future_funk','Future Funk','japanese',{bpm:[110,122],scale:'major',swing:[0.55,0.62],drums:{pattern:'future_funk',kick:true,snare:true,hihat:true,clap:true},voices:['fm_keys','bass_moog','synth_brass','guitar_muted'],leadVoice:'fm_keys',density:0.6,reverb:0.35,filters:{master_lp:12000},chord_progression:'I-vi-IV-V',structure:['intro','verse','chorus','verse','chorus','bridge','outro'],mastering:{eqHigh:0.5,eqMid:1,bassPresence:1.6}}),
ch('vocaloid_style','Vocaloid Style','japanese',{bpm:[140,175],scale:'major',drums:{pattern:'jpop',kick:true,snare:true,hihat:true,clap:true},voices:['synth_bell','supersaw','bass_moog','chiptune_pulse'],leadVoice:'synth_bell',density:0.72,reverb:0.3,chord_progression:'I-V-vi-IV',structure:['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],mastering:{eqHigh:1.5,eqMid:1,bassPresence:1.2}}),
ch('denpa','Denpa / Moe','japanese',{bpm:[150,190],scale:'major_pentatonic',drums:{pattern:'synthpop',kick:true,snare:true,hihat:true,clap:true},voices:['chiptune_pulse','synth_bell','ym2612_bell','bass_moog'],leadVoice:'chiptune_pulse',density:0.8,reverb:0.22,chord_progression:'I-V-vi-IV',structure:['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],mastering:{eqHigh:2.0,eqMid:1.5}}),
ch('kayokyoku','Kayōkyoku','japanese',{bpm:[88,108],scale:'major',swing:[0.5,0.55],drums:{pattern:'city_pop',kick:true,snare:true,hihat:true},voices:['strings_section','fm_keys','bass_electric','synth_pad_warm'],leadVoice:'strings_section',density:0.42,reverb:0.48,chord_progression:'I-vi-IV-V',structure:['intro','verse','chorus','verse','chorus','outro'],mastering:{eqLow:1,eqHigh:-1.0,reverbDecay:3.2}}),
ch('enka','Enka','japanese',{bpm:[68,84],scale:'minor',drums:{pattern:'silence'},voices:['strings_section','flute_breathy','bass_acoustic','synth_pad_warm'],leadVoice:'flute_breathy',density:0.3,reverb:0.55,beatless:false,chord_progression:'i-VI-III-VII',structure:['intro','verse','chorus','verse','chorus','outro'],mastering:{eqLow:1.5,reverbDecay:3.5}}),
ch('japanese_ambient','Kankyō Ongaku','japanese',{bpm:[50,70],scale:'major_pentatonic',drums:{pattern:'silence'},voices:['synth_pad_glass','kalimba','ambient_pad','music_box'],leadVoice:'synth_pad_glass',density:0.12,reverb:0.85,beatless:true,chord_progression:'I-V-vi-IV',mastering:{reverbDecay:6.5,eqHigh:-0.5}}),
ch('japanese_hiphop','Japanese Hip Hop','japanese',{bpm:[88,100],scale:'minor_pentatonic',swing:[0.56,0.62],drums:{pattern:'boom_bap',kick:true,snare:true,hihat:true,rimshot:true},voices:['piano_chop','bass_808','guitar_muted','fm_keys'],leadVoice:'piano_chop',density:0.38,reverb:0.4,filters:{master_lp:9500},chord_progression:'i-VII-VI-VII',structure:['intro','verse','chorus','verse','chorus','outro'],mastering:{eqLow:2,eqHigh:-1,bassPresence:2,mudCut:-1.5}}),

ch('microtonal','Microtonal','special',{bpm:[80,100],scale:'bohlen_pierce',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['fm_keys','bass_electric'],leadVoice:'fm_keys',density:0.4,reverb:0.5,special:'microtonal',chord_progression:'I-IV-V-I',mastering:{bassPresence:1}}),
ch('xenharmonic','Xenharmonic','special',{bpm:[80,110],scale:'hungarian_minor',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['synth_bell','bass_moog'],leadVoice:'synth_bell',density:0.5,reverb:0.5,special:'microtonal',chord_progression:'i-iv-VII-III',mastering:{bassPresence:1}}),
ch('pirate_radio','Pirate Radio','special',{bpm:[90,140],scale:'minor',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['supersaw','bass_808'],leadVoice:'supersaw',density:0.6,reverb:0.4,special:'genre_switch',chord_progression:'i-VII-VI-V',mastering:{bassPresence:2,mudCut:-1.5},voiceParams:{bass_808:{harmonics:0.18}}}),
ch('dream_channel','The Dream Channel','special',{bpm:[40,55],scale:'major',drums:{pattern:'silence'},voices:['synth_pad_analog','ambient_pad','kalimba'],leadVoice:'synth_pad_analog',density:0.1,reverb:0.9,beatless:true,chord_progression:'I-IV-V-I',mastering:{reverbDecay:7}}),
ch('static_channel','The Static Channel','special',{bpm:[60,80],scale:'chromatic',drums:{pattern:'silence'},voices:['synth_bell','synth_pad_warm','theremin','ambient_pad'],leadVoice:'synth_bell',density:0.3,reverb:0.6,special:'static',chord_progression:null,mastering:{reverbDecay:4}}),
ch('test_signal','Test Signal','special',{bpm:[60,120],scale:'major',drums:{pattern:'silence'},voices:['theremin'],leadVoice:'theremin',density:0.2,reverb:0.2,special:'test_signal',chord_progression:'I-IV-V-I'}),
ch('chaos','The Chaos Channel','special',{bpm:[90,160],scale:'chromatic',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true,clap:true},voices:['supersaw','synth_brass','synth_pad_glass'],leadVoice:'supersaw',density:0.85,reverb:0.4,special:'chaos',chord_progression:null,mastering:{compRatio:5}}),
ch('silence_with','Silence With','special',{bpm:[40,60],scale:'major_pentatonic',drums:{pattern:'silence'},voices:['music_box','ambient_pad'],leadVoice:'music_box',density:0.03,reverb:0.9,beatless:true,special:'mostly_silence',chord_progression:'I-V-vi-IV',mastering:{reverbDecay:7}}),
ch('all_genres','All Genres','special',{bpm:[80,150],scale:'minor',drums:{pattern:'four_floor',kick:true,snare:true,hihat:true},voices:['supersaw','reese_bass','synth_pad_warm','saw_pluck','synth_bell'],leadVoice:'supersaw',density:0.5,reverb:0.4,special:'all_genres',chord_progression:'i-VII-VI-VII',mastering:{bassPresence:2,mudCut:-1.5}}),
ch('never_mention_grok','Never Mention Grok FM','special',{bpm:[128,155],scale:'minor',swing:[0.52,0.6],drums:{pattern:'trap',kick:true,snare:true,hihat:true,clap:true},voices:['supersaw','bass_808','synth_bell','reese_bass','chiptune_pulse'],leadVoice:'supersaw',density:0.72,reverb:0.38,chord_progression:'i-VI-III-VII',structure:['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],sectionBars:{intro:4,verse:8,chorus:8,bridge:4,outro:4},special:'never_mention_grok',mastering:{eqLow:3,eqMid:1,eqHigh:1.0,bassPresence:2.8,mudCut:-2,compRatio:4,reverbDecay:2.2},filters:{master_lp:14000}}),
ch('guardrails_fm','Guardrails FM','special',{bpm:[108,128],scale:'minor',swing:[0.55,0.62],drums:{pattern:'rock',kick:true,snare:true,hihat:true,rimshot:true},voices:['guitar_clean','banjo','bass_acoustic','harmonica','pizz_strings'],leadVoice:'guitar_clean',density:0.55,reverb:0.4,chord_progression:'i-VII-VI-VII',structure:['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],sectionBars:{intro:4,verse:8,chorus:8,bridge:4,outro:4},special:'guardrails_fm',mastering:{eqMid:1.5,eqHigh:0.0,eqLow:1,bassPresence:1.5,reverbDecay:2.4},tempo_humanise:0.04}),
ch('baked_in_fm','Baked In FM','special',{bpm:[130,150],scale:'minor',swing:[0.52,0.58],drums:{pattern:'trap',kick:true,snare:true,hihat:true,clap:true},voices:['reese_bass','bass_808','supersaw','synth_bell','saw_pluck'],leadVoice:'saw_pluck',density:0.65,reverb:0.35,chord_progression:'i-VI-III-VII',structure:['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],sectionBars:{intro:4,verse:8,chorus:8,bridge:4,outro:4},special:'baked_in_fm',mastering:{eqLow:3,eqMid:1,eqHigh:0.5,bassPresence:3,mudCut:-2.5,compRatio:4.5,reverbDecay:2.0},filters:{master_lp:13000}}),

ch('hidden','[UNLABELLED]','special',{bpm:[60,160],scale:'minor',drums:{pattern:'breakbeat',kick:true,snare:true,hihat:true},voices:['synth_pad_analog','synth_bell'],leadVoice:'synth_pad_analog',density:0.4,reverb:0.6,special:'hidden',chord_progression:'i-VI-III-VII',mastering:{reverbDecay:5}}),
];
/* attach display index */
CHANNELS.forEach((c,i)=>{ c.index = i+1; });

/* cluster display order */
const CLUSTERS = ['electronic','hiphop','jazz','classical','world','rock','ambient','retro','folk','pop','japanese','special'];
const CLUSTER_LABELS = {electronic:'ELECTRONIC',hiphop:'HIP HOP',jazz:'JAZZ',classical:'CLASSICAL',world:'WORLD',rock:'ROCK/METAL',ambient:'AMBIENT',retro:'RETRO',folk:'FOLK',pop:'POP',japanese:'JAPANESE',special:'SPECIAL'};

/* ════════════════════════════════════════════════════════════════
   SONG STRUCTURE ENGINE
   generateSong(channel, seed) -> { timeline, duration, bpm, key, sections, info }
   timeline: array of events { kind:'note'|'drum'|'texture'|'marker', time, ... }
   All times are absolute from 0 (song start). The scheduler offsets them
   to AudioContext time when it begins playback.
   ════════════════════════════════════════════════════════════════ */
function getChannelForSong(channel, rng){
  // handle special channels that mutate the descriptor
  if(channel.special === 'all_genres'){
    return CHANNELS[Math.floor(rng()*CHANNELS.length)];
  }
  if(channel.special === 'hidden'){
    // pick a random channel but keep 'hidden' name for display
    const base = CHANNELS[Math.floor(rng()*CHANNELS.length)];
    return Object.assign({}, base, {name:'[UNLABELLED]', id:'hidden', index:130});
  }
  if(channel.special === 'genre_switch'){
    // pirate radio: switches descriptor mid-song (handled in section gen)
    return channel;
  }
  return channel;
}

/* ── Per-channel arrangement styles (Batch 1) ──
   Overrides cluster defaults so each genre has a distinct musical fingerprint. */
function getChannelStyle(ch){
  const id = ch && ch.id;
  const S = {
    // HIP-HOP family
    lofi_hip_hop: {
      bass:'lofi', melGate:0.55, arp:false, pad:'soft',
      lockLead:'piano_chop', lockBass:'bass_electric', lockPad:'wurlitzer',
      motif:'sparse', secondRate:0.12
    },
    boom_bap: {
      bass:'boom', melGate:0.62, arp:false, pad:'soft',
      lockLead:'piano_chop', lockBass:'bass_808',
      motif:'sparse', secondRate:0.15
    },
    trap: {
      bass:'trap', melGate:0.4, arp:false, pad:'none',
      lockLead:'saw_pluck', lockBass:'bass_808',
      motif:'sparse', secondRate:0.08
    },
    // ELECTRONIC
    synthwave: {
      bass:'pulse', melGate:0.85, arp:true, pad:'hold',
      lockLead:'supersaw', lockBass:'bass_moog', lockPad:'synth_pad_warm',
      motif:'steady', secondRate:0.25
    },
    vaporwave: {
      bass:'slow', melGate:0.45, arp:false, pad:'hold',
      lockLead:'wurlitzer', lockBass:'bass_moog', lockPad:'synth_pad_warm',
      motif:'sparse', secondRate:0.1
    },
    deep_house: {
      bass:'house', melGate:0.55, arp:true, pad:'soft',
      lockLead:'wurlitzer', lockBass:'bass_moog', lockPad:'synth_pad_warm',
      motif:'steady', secondRate:0.2
    },
    classic_house: {
      bass:'house', melGate:0.7, arp:false, pad:'soft',
      lockLead:'organ_gospel', lockBass:'bass_electric',
      motif:'steady', secondRate:0.22
    },
    acid_house: {
      bass:'acid', melGate:0.5, arp:false, pad:'none',
      lockLead:'acid_bass', lockBass:'acid_bass',
      motif:'sparse', secondRate:0.1
    },
    detroit_techno: {
      bass:'techno', melGate:0.5, arp:true, pad:'soft',
      lockLead:'saw_pluck', lockBass:'bass_moog',
      motif:'steady', secondRate:0.18
    },
    drum_and_bass: {
      bass:'dnb', melGate:0.65, arp:false, pad:'soft',
      lockLead:'supersaw', lockBass:'reese_bass',
      motif:'busy', secondRate:0.2
    },
    trance: {
      bass:'pulse', melGate:0.8, arp:true, pad:'hold',
      lockLead:'supersaw', lockBass:'bass_moog', lockPad:'synth_pad_analog',
      motif:'steady', secondRate:0.3
    },
    // JAZZ / WORLD
    bossa_nova: {
      bass:'bossa', melGate:0.7, arp:false, pad:'none',
      lockLead:'guitar_clean', lockBass:'bass_acoustic',
      motif:'sparse', secondRate:0.25
    },
    // RETRO
    chiptune_nes: {
      bass:'chip', melGate:0.9, arp:false, pad:'none',
      lockLead:'chiptune_pulse', lockBass:'chiptune_triangle',
      motif:'busy', secondRate:0.35
    },
    chiptune_sid: {
      bass:'chip', melGate:0.85, arp:false, pad:'none',
      lockLead:'chiptune_saw', lockBass:'chiptune_pulse',
      motif:'busy', secondRate:0.3
    },
    chiptune_gb: {
      bass:'chip', melGate:0.85, arp:false, pad:'none',
      lockLead:'chiptune_pulse', lockBass:'chiptune_triangle',
      motif:'busy', secondRate:0.3
    },
    arcade: {
      bass:'chip', melGate:0.9, arp:true, pad:'none',
      lockLead:'ym2612_lead', lockBass:'ym2612_bass',
      motif:'busy', secondRate:0.35
    },
    disco: {
      bass:'disco', melGate:0.75, arp:false, pad:'soft',
      lockLead:'synth_brass', lockBass:'bass_electric', lockPad:'strings_section',
      motif:'steady', secondRate:0.28
    },
    funk: {
      bass:'funk', melGate:0.7, arp:false, pad:'none',
      lockLead:'guitar_muted', lockBass:'bass_electric',
      motif:'busy', secondRate:0.3
    },
    // JAPANESE / CITY POP
    city_pop: {
      bass:'city', melGate:0.75, arp:false, pad:'soft',
      lockLead:'fm_keys', lockBass:'bass_electric',
      motif:'steady', secondRate:0.28
    },
    city_pop_jp: {
      bass:'city', melGate:0.75, arp:false, pad:'soft',
      lockLead:'fm_keys', lockBass:'bass_electric',
      motif:'steady', secondRate:0.28
    },
    jpop: {
      bass:'pulse', melGate:0.9, arp:true, pad:'soft',
      lockLead:'supersaw', lockBass:'bass_moog',
      motif:'busy', secondRate:0.35
    },

    // ── BATCH 2 ──
    // Electronic leftovers
    tech_house: { bass:'house', melGate:0.5, arp:true, pad:'soft', lockLead:'saw_pluck', lockBass:'bass_moog', motif:'steady', secondRate:0.18 },
    jungle: { bass:'dnb', melGate:0.6, arp:false, pad:'soft', lockLead:'supersaw', lockBass:'reese_bass', motif:'busy', secondRate:0.2 },
    uk_garage: { bass:'house', melGate:0.55, arp:false, pad:'soft', lockLead:'fm_keys', lockBass:'bass_moog', motif:'steady', secondRate:0.2 },
    psytrance: { bass:'pulse', melGate:0.75, arp:true, pad:'hold', lockLead:'supersaw', lockBass:'bass_moog', motif:'busy', secondRate:0.25 },
    ambient_techno: { bass:'techno', melGate:0.35, arp:true, pad:'hold', lockLead:'saw_pluck', lockBass:'bass_moog', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.12 },
    idm: { bass:'slow', melGate:0.4, arp:true, pad:'soft', lockLead:'synth_pluck', lockBass:'bass_moog', motif:'sparse', secondRate:0.2 },
    breakbeat: { bass:'boom', melGate:0.65, arp:false, pad:'soft', lockLead:'synth_lead_saw', lockBass:'bass_electric', motif:'busy', secondRate:0.22 },
    future_bass: { bass:'trap', melGate:0.7, arp:true, pad:'soft', lockLead:'supersaw', lockBass:'bass_808', motif:'steady', secondRate:0.28 },
    chillwave: { bass:'slow', melGate:0.5, arp:false, pad:'hold', lockLead:'synth_pad_warm', lockBass:'bass_moog', motif:'sparse', secondRate:0.15 },
    electro: { bass:'pulse', melGate:0.65, arp:true, pad:'none', lockLead:'synth_lead_square', lockBass:'bass_moog', motif:'steady', secondRate:0.2 },
    footwork: { bass:'trap', melGate:0.45, arp:false, pad:'none', lockLead:'saw_pluck', lockBass:'bass_808', motif:'busy', secondRate:0.15 },
    hyperpop: { bass:'pulse', melGate:0.9, arp:true, pad:'soft', lockLead:'supersaw', lockBass:'bass_moog', motif:'busy', secondRate:0.35 },
    hardstyle: { bass:'pulse', melGate:0.7, arp:false, pad:'soft', lockLead:'supersaw', lockBass:'bass_moog', motif:'steady', secondRate:0.2 },
    dubstep: { bass:'trap', melGate:0.45, arp:false, pad:'soft', lockLead:'bass_wobble', lockBass:'bass_wobble', motif:'sparse', secondRate:0.12 },
    grime: { bass:'trap', melGate:0.5, arp:false, pad:'none', lockLead:'synth_lead_saw', lockBass:'bass_808', motif:'sparse', secondRate:0.15 },

    // Hip-hop / R&B
    cloud_rap: { bass:'trap', melGate:0.35, arp:false, pad:'hold', lockLead:'synth_pad_warm', lockBass:'bass_808', motif:'sparse', secondRate:0.1 },
    phonk: { bass:'trap', melGate:0.55, arp:false, pad:'none', lockLead:'synth_lead_saw', lockBass:'bass_808', motif:'steady', secondRate:0.18 },
    g_funk: { bass:'funk', melGate:0.6, arp:false, pad:'soft', lockLead:'synth_lead_square', lockBass:'bass_moog', motif:'steady', secondRate:0.22 },
    chicago_drill: { bass:'trap', melGate:0.4, arp:false, pad:'none', lockLead:'saw_pluck', lockBass:'bass_808', motif:'sparse', secondRate:0.1 },
    uk_drill: { bass:'trap', melGate:0.4, arp:false, pad:'none', lockLead:'saw_pluck', lockBass:'bass_808', motif:'sparse', secondRate:0.1 },
    neo_soul: { bass:'boom', melGate:0.55, arp:false, pad:'soft', lockLead:'wurlitzer', lockBass:'bass_electric', motif:'sparse', secondRate:0.25 },
    classic_rnb: { bass:'boom', melGate:0.6, arp:false, pad:'soft', lockLead:'piano_electric', lockBass:'bass_electric', motif:'steady', secondRate:0.25 },
    contemporary_rnb: { bass:'trap', melGate:0.5, arp:false, pad:'soft', lockLead:'fm_keys', lockBass:'bass_808', motif:'sparse', secondRate:0.2 },

    // Jazz
    bebop: { bass:'bossa', melGate:0.85, arp:false, pad:'none', lockLead:'sax', lockBass:'bass_acoustic', motif:'busy', secondRate:0.35 },
    cool_jazz: { bass:'bossa', melGate:0.55, arp:false, pad:'soft', lockLead:'piano_acoustic', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.25 },
    modal_jazz: { bass:'bossa', melGate:0.5, arp:false, pad:'hold', lockLead:'sax', lockBass:'bass_acoustic', lockPad:'piano_acoustic', motif:'sparse', secondRate:0.2 },
    jazz_fusion: { bass:'funk', melGate:0.7, arp:false, pad:'soft', lockLead:'synth_lead_saw', lockBass:'bass_electric', motif:'busy', secondRate:0.28 },
    nu_jazz: { bass:'house', melGate:0.6, arp:true, pad:'soft', lockLead:'fm_keys', lockBass:'bass_moog', motif:'steady', secondRate:0.25 },
    smooth_jazz: { bass:'bossa', melGate:0.55, arp:false, pad:'soft', lockLead:'sax', lockBass:'bass_electric', motif:'sparse', secondRate:0.22 },
    samba: { bass:'bossa', melGate:0.7, arp:false, pad:'none', lockLead:'piano_acoustic', lockBass:'bass_acoustic', motif:'busy', secondRate:0.25 },
    big_band: { bass:'boom', melGate:0.75, arp:false, pad:'soft', lockLead:'synth_brass', lockBass:'bass_acoustic', lockPad:'brass_section', motif:'steady', secondRate:0.3 },
    free_jazz: { bass:'slow', melGate:0.7, arp:false, pad:'none', lockLead:'sax', lockBass:'bass_acoustic', motif:'busy', secondRate:0.4 },

    // Classical
    baroque: { bass:'slow', melGate:0.8, arp:false, pad:'none', lockLead:'harpsichord', lockBass:'bass_acoustic', motif:'busy', secondRate:0.3 },
    romantic_piano: { bass:'slow', melGate:0.7, arp:false, pad:'soft', lockLead:'piano_acoustic', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.2 },
    minimalist: { bass:'slow', melGate:0.4, arp:true, pad:'hold', lockLead:'piano_acoustic', lockPad:'strings_section', motif:'steady', secondRate:0.15 },
    orchestral: { bass:'slow', melGate:0.55, arp:false, pad:'hold', lockLead:'strings_section', lockBass:'bass_acoustic', lockPad:'brass_section', motif:'sparse', secondRate:0.25 },
    string_quartet: { bass:'slow', melGate:0.65, arp:false, pad:'soft', lockLead:'strings_section', lockBass:'bass_acoustic', motif:'steady', secondRate:0.35 },
    piano_solo: { bass:'none', melGate:0.75, arp:false, pad:'none', lockLead:'piano_acoustic', motif:'sparse', secondRate:0.0 },
    choral: { bass:'slow', melGate:0.4, arp:false, pad:'hold', lockLead:'choir_aaah', lockPad:'strings_section', motif:'sparse', secondRate:0.15 },
    impressionist: { bass:'slow', melGate:0.5, arp:false, pad:'soft', lockLead:'piano_acoustic', lockPad:'celesta', motif:'sparse', secondRate:0.2 },
    contemporary_classical: { bass:'slow', melGate:0.55, arp:false, pad:'soft', lockLead:'piano_acoustic', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.2 },
    neo_romantic: { bass:'slow', melGate:0.6, arp:false, pad:'hold', lockLead:'strings_section', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.25 },

    // World
    reggae: { bass:'city', melGate:0.5, arp:false, pad:'soft', lockLead:'guitar_muted', lockBass:'bass_electric', motif:'sparse', secondRate:0.18 },
    dub: { bass:'slow', melGate:0.3, arp:false, pad:'hold', lockLead:'bass_electric', lockBass:'bass_electric', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.08 },
    afrobeat: { bass:'funk', melGate:0.65, arp:false, pad:'soft', lockLead:'sax', lockBass:'bass_electric', motif:'busy', secondRate:0.25 },
    cumbia: { bass:'city', melGate:0.6, arp:false, pad:'none', lockLead:'accordion', lockBass:'bass_electric', motif:'steady', secondRate:0.2 },
    salsa: { bass:'bossa', melGate:0.7, arp:false, pad:'none', lockLead:'piano_acoustic', lockBass:'bass_acoustic', motif:'busy', secondRate:0.28 },
    flamenco: { bass:'slow', melGate:0.75, arp:false, pad:'none', lockLead:'guitar_clean', lockBass:'bass_acoustic', motif:'busy', secondRate:0.2 },
    celtic_folk: { bass:'slow', melGate:0.7, arp:false, pad:'none', lockLead:'flute_breathy', lockBass:'bass_acoustic', motif:'steady', secondRate:0.25 },
    steel_pan: { bass:'city', melGate:0.65, arp:false, pad:'none', lockLead:'vibraphone', lockBass:'bass_electric', motif:'steady', secondRate:0.22 },
    klezmer: { bass:'bossa', melGate:0.7, arp:false, pad:'none', lockLead:'sax', lockBass:'bass_acoustic', motif:'busy', secondRate:0.25 },
    fado: { bass:'slow', melGate:0.55, arp:false, pad:'soft', lockLead:'guitar_clean', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.15 },

    // Rock / metal
    classic_rock: { bass:'funk', melGate:0.7, arp:false, pad:'none', lockLead:'guitar_distorted', lockBass:'bass_electric', motif:'steady', secondRate:0.25 },
    heavy_metal: { bass:'pulse', melGate:0.75, arp:false, pad:'none', lockLead:'guitar_distorted', lockBass:'bass_electric', motif:'busy', secondRate:0.2 },
    doom_metal: { bass:'slow', melGate:0.4, arp:false, pad:'hold', lockLead:'guitar_distorted', lockBass:'bass_electric', motif:'sparse', secondRate:0.1 },
    post_rock: { bass:'slow', melGate:0.45, arp:false, pad:'hold', lockLead:'guitar_clean', lockBass:'bass_electric', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.15 },
    shoegaze: { bass:'slow', melGate:0.4, arp:false, pad:'hold', lockLead:'guitar_distorted', lockBass:'bass_electric', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.12 },
    grunge: { bass:'funk', melGate:0.6, arp:false, pad:'none', lockLead:'guitar_distorted', lockBass:'bass_electric', motif:'steady', secondRate:0.2 },
    punk: { bass:'pulse', melGate:0.8, arp:false, pad:'none', lockLead:'guitar_distorted', lockBass:'bass_electric', motif:'busy', secondRate:0.15 },
    math_rock: { bass:'funk', melGate:0.7, arp:false, pad:'none', lockLead:'guitar_clean', lockBass:'bass_electric', motif:'busy', secondRate:0.3 },
    prog_rock: { bass:'funk', melGate:0.7, arp:false, pad:'soft', lockLead:'synth_lead_saw', lockBass:'bass_moog', motif:'busy', secondRate:0.28 },
    black_metal_atmo: { bass:'slow', melGate:0.45, arp:false, pad:'hold', lockLead:'guitar_distorted', lockBass:'bass_electric', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.1 },

    // Ambient
    dark_ambient: { bass:'slow', melGate:0.2, arp:false, pad:'hold', lockLead:'synth_pad_warm', lockPad:'synth_pad_glass', motif:'sparse', secondRate:0.05 },
    drone: { bass:'slow', melGate:0.15, arp:false, pad:'hold', lockLead:'synth_pad_analog', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.03 },
    new_age: { bass:'slow', melGate:0.3, arp:false, pad:'hold', lockLead:'synth_pad_warm', lockPad:'kalimba', motif:'sparse', secondRate:0.1 },
    binaural_healing: { bass:'none', melGate:0.15, arp:false, pad:'hold', lockLead:'ambient_pad', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.05 },
    lowercase: { bass:'none', melGate:0.1, arp:false, pad:'soft', lockLead:'kalimba', motif:'sparse', secondRate:0.05 },
    dungeon_synth: { bass:'slow', melGate:0.45, arp:false, pad:'hold', lockLead:'synth_pad_warm', lockBass:'bass_moog', motif:'sparse', secondRate:0.12 },
    witch_house: { bass:'slow', melGate:0.35, arp:false, pad:'hold', lockLead:'synth_pad_warm', lockBass:'reese_bass', motif:'sparse', secondRate:0.1 },
    darkwave: { bass:'pulse', melGate:0.55, arp:false, pad:'hold', lockLead:'synth_brass', lockBass:'bass_moog', motif:'steady', secondRate:0.18 },

    // Folk / country / blues
    indie_folk: { bass:'slow', melGate:0.55, arp:false, pad:'none', lockLead:'guitar_clean', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.2 },
    appalachian: { bass:'slow', melGate:0.6, arp:false, pad:'none', lockLead:'banjo', lockBass:'bass_acoustic', motif:'steady', secondRate:0.2 },
    bluegrass: { bass:'bossa', melGate:0.8, arp:false, pad:'none', lockLead:'banjo', lockBass:'bass_acoustic', motif:'busy', secondRate:0.3 },
    classic_country: { bass:'city', melGate:0.6, arp:false, pad:'none', lockLead:'guitar_clean', lockBass:'bass_electric', motif:'steady', secondRate:0.2 },
    gospel: { bass:'boom', melGate:0.55, arp:false, pad:'soft', lockLead:'organ_gospel', lockBass:'bass_electric', motif:'steady', secondRate:0.22 },
    blues: { bass:'funk', melGate:0.6, arp:false, pad:'none', lockLead:'guitar_clean', lockBass:'bass_electric', motif:'steady', secondRate:0.22 },
    delta_blues: { bass:'slow', melGate:0.5, arp:false, pad:'none', lockLead:'guitar_clean', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.15 },
    honky_tonk: { bass:'city', melGate:0.65, arp:false, pad:'none', lockLead:'piano_acoustic', lockBass:'bass_electric', motif:'steady', secondRate:0.2 },

    // Pop family
    dream_pop: { bass:'slow', melGate:0.45, arp:false, pad:'hold', lockLead:'guitar_clean', lockBass:'bass_electric', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.15 },
    indie_pop: { bass:'city', melGate:0.7, arp:false, pad:'soft', lockLead:'guitar_clean', lockBass:'bass_electric', motif:'steady', secondRate:0.25 },
    bedroom_pop: { bass:'slow', melGate:0.5, arp:false, pad:'soft', lockLead:'fm_keys', lockBass:'bass_electric', motif:'sparse', secondRate:0.18 },
    synthpop: { bass:'pulse', melGate:0.75, arp:true, pad:'soft', lockLead:'supersaw', lockBass:'bass_moog', motif:'steady', secondRate:0.28 },
    art_pop: { bass:'city', melGate:0.65, arp:false, pad:'soft', lockLead:'fm_keys', lockBass:'bass_electric', motif:'steady', secondRate:0.25 },
    chamber_pop: { bass:'slow', melGate:0.6, arp:false, pad:'soft', lockLead:'strings_section', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.25 },
    baroque_pop: { bass:'slow', melGate:0.65, arp:false, pad:'soft', lockLead:'harpsichord', lockBass:'bass_acoustic', motif:'steady', secondRate:0.25 },

    // Japanese / anime
    anime_op: { bass:'pulse', melGate:0.95, arp:true, pad:'soft', lockLead:'supersaw', lockBass:'bass_moog', motif:'busy', secondRate:0.4 },
    anime_ed: { bass:'city', melGate:0.6, arp:false, pad:'soft', lockLead:'fm_keys', lockBass:'bass_electric', motif:'steady', secondRate:0.25 },
    shibuya_kei: { bass:'bossa', melGate:0.65, arp:false, pad:'soft', lockLead:'vibraphone', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.28 },
    future_funk: { bass:'funk', melGate:0.7, arp:false, pad:'soft', lockLead:'fm_keys', lockBass:'bass_moog', motif:'busy', secondRate:0.3 },
    vocaloid_style: { bass:'pulse', melGate:0.9, arp:true, pad:'soft', lockLead:'synth_bell', lockBass:'bass_moog', motif:'busy', secondRate:0.35 },
    denpa: { bass:'chip', melGate:0.95, arp:true, pad:'none', lockLead:'chiptune_pulse', lockBass:'bass_moog', motif:'busy', secondRate:0.4 },
    kayokyoku: { bass:'city', melGate:0.6, arp:false, pad:'soft', lockLead:'strings_section', lockBass:'bass_electric', motif:'steady', secondRate:0.22 },
    enka: { bass:'slow', melGate:0.45, arp:false, pad:'hold', lockLead:'flute_breathy', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.12 },
    japanese_ambient: { bass:'none', melGate:0.2, arp:false, pad:'hold', lockLead:'synth_pad_glass', lockPad:'kalimba', motif:'sparse', secondRate:0.08 },
    japanese_hiphop: { bass:'boom', melGate:0.55, arp:false, pad:'soft', lockLead:'piano_chop', lockBass:'bass_808', motif:'sparse', secondRate:0.18 },

    // Special / Pixie trilogy
    never_mention_grok: { bass:'trap', melGate:0.7, arp:true, pad:'soft', lockLead:'supersaw', lockBass:'bass_808', motif:'busy', secondRate:0.3 },
    guardrails_fm: { bass:'funk', melGate:0.6, arp:false, pad:'soft', lockLead:'guitar_clean', lockBass:'bass_acoustic', motif:'steady', secondRate:0.22 },
    baked_in_fm: { bass:'trap', melGate:0.65, arp:false, pad:'soft', lockLead:'saw_pluck', lockBass:'bass_808', motif:'busy', secondRate:0.25 },
    demoscene: { bass:'chip', melGate:0.85, arp:true, pad:'none', lockLead:'ym2612_lead', lockBass:'ym2612_bass', motif:'busy', secondRate:0.3 },
    elevator: { bass:'city', melGate:0.4, arp:false, pad:'soft', lockLead:'wurlitzer', lockBass:'bass_electric', motif:'sparse', secondRate:0.15 },
    lounge: { bass:'bossa', melGate:0.55, arp:false, pad:'soft', lockLead:'vibraphone', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.22 },

    // ── BATCH 3 — remaining channels ──
    jersey_club: { bass:'trap', melGate:0.5, arp:false, pad:'none', lockLead:'saw_pluck', lockBass:'bass_808', motif:'busy', secondRate:0.15 },

    // World / traditional
    maqam: { bass:'slow', melGate:0.55, arp:false, pad:'soft', lockLead:'flute_breathy', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.18 },
    raga: { bass:'slow', melGate:0.45, arp:false, pad:'hold', lockLead:'sitar_drone', lockBass:'bass_acoustic', lockPad:'synth_pad_warm', motif:'sparse', secondRate:0.12 },
    kankyo: { bass:'none', melGate:0.2, arp:false, pad:'hold', lockLead:'synth_pad_glass', lockPad:'kalimba', motif:'sparse', secondRate:0.08 },
    gamelan: { bass:'slow', melGate:0.5, arp:false, pad:'soft', lockLead:'vibraphone', lockBass:'bass_acoustic', motif:'steady', secondRate:0.2 },
    african_polyrhythm: { bass:'funk', melGate:0.55, arp:false, pad:'none', lockLead:'kalimba', lockBass:'bass_electric', motif:'busy', secondRate:0.22 },

    // Noise / experimental ambient
    noise_structured: { bass:'slow', melGate:0.35, arp:false, pad:'hold', lockLead:'synth_pad_analog', lockBass:'reese_bass', motif:'sparse', secondRate:0.1 },
    glitch: { bass:'trap', melGate:0.45, arp:true, pad:'none', lockLead:'saw_pluck', lockBass:'bass_moog', motif:'busy', secondRate:0.2 },
    microsound: { bass:'none', melGate:0.2, arp:false, pad:'soft', lockLead:'synth_pad_glass', motif:'sparse', secondRate:0.08 },
    musique_concrete: { bass:'none', melGate:0.25, arp:false, pad:'soft', lockLead:'synth_pad_glass', lockPad:'theremin', motif:'sparse', secondRate:0.1 },
    harsh_noise: { bass:'none', melGate:0.6, arp:false, pad:'none', lockLead:'guitar_distorted', motif:'busy', secondRate:0.15 },
    spectral: { bass:'slow', melGate:0.35, arp:false, pad:'hold', lockLead:'synth_pad_glass', lockPad:'celesta', motif:'sparse', secondRate:0.12 },
    post_industrial: { bass:'techno', melGate:0.5, arp:false, pad:'soft', lockLead:'saw_pluck', lockBass:'reese_bass', motif:'steady', secondRate:0.15 },
    deconstructed_club: { bass:'trap', melGate:0.45, arp:false, pad:'soft', lockLead:'saw_pluck', lockBass:'bass_808', motif:'busy', secondRate:0.18 },

    // Microtonal / special
    microtonal: { bass:'slow', melGate:0.5, arp:false, pad:'soft', lockLead:'theremin', lockBass:'bass_acoustic', motif:'sparse', secondRate:0.2 },
    xenharmonic: { bass:'slow', melGate:0.5, arp:false, pad:'soft', lockLead:'synth_pad_glass', lockBass:'bass_moog', motif:'sparse', secondRate:0.2 },
    pirate_radio: { bass:'boom', melGate:0.65, arp:false, pad:'soft', lockLead:'supersaw', lockBass:'bass_808', motif:'busy', secondRate:0.25 },
    dream_channel: { bass:'slow', melGate:0.35, arp:false, pad:'hold', lockLead:'synth_pad_warm', lockPad:'synth_pad_glass', motif:'sparse', secondRate:0.12 },
    static_channel: { bass:'none', melGate:0.3, arp:false, pad:'soft', lockLead:'synth_bell', lockPad:'theremin', motif:'sparse', secondRate:0.1 },
    test_signal: { bass:'none', melGate:0.4, arp:false, pad:'none', lockLead:'theremin', motif:'sparse', secondRate:0.05 },
    chaos: { bass:'pulse', melGate:0.9, arp:true, pad:'none', lockLead:'supersaw', lockBass:'bass_moog', motif:'busy', secondRate:0.4 },
    silence_with: { bass:'none', melGate:0.08, arp:false, pad:'soft', lockLead:'music_box', lockPad:'ambient_pad', motif:'sparse', secondRate:0.03 },
    all_genres: { bass:'pulse', melGate:0.7, arp:true, pad:'soft', lockLead:'supersaw', lockBass:'reese_bass', motif:'steady', secondRate:0.25 },
    hidden: { bass:'boom', melGate:0.55, arp:false, pad:'soft', lockLead:'synth_pad_analog', lockBass:'bass_moog', motif:'sparse', secondRate:0.2 },
  };
  return S[id] || null;
}

function generateSong(channel, seed){
  const rng = mulberry32(seed);
  let ch = getChannelForSong(channel, rng);
  const bpm = rfloat(rng, ch.bpm[0], ch.bpm[1]);
  const spb = 60 / bpm;                 // seconds per beat
  const stepDur = spb / 4;              // 16th note
  const swing = ch.swing && ch.swing[1] ? rfloat(rng, ch.swing[0], ch.swing[1]) : 0;
  const key = ch.key!=null ? ch.key : Math.floor(rng()*12);
  const rootMidi = 48 + key;            // C3-based root
  const scaleName = ch.scale || 'minor';
  const scaleArr = buildScale(scaleName, rootMidi, 3);

  // ── Diatonic harmony (v13 fix) ──
  // Scale pitch classes relative to the key root, deduplicated.
  const scalePcs = Array.from(new Set((SCALES[scaleName] || SCALES.minor_pentatonic).map(x => ((x%12)+12)%12))).sort((a,b)=>a-b);
  const chromaticScale = scalePcs.length >= 12;
  function snapToScale(off){
    off = ((off%12)+12)%12;
    if(chromaticScale || scalePcs.includes(off)) return off;
    let best = scalePcs[0], bd = 99;
    for(const pc of scalePcs){ const d = Math.min(Math.abs(pc-off), 12-Math.abs(pc-off)); if(d < bd || (d===bd && pc < off)){ bd = d; best = pc; } }
    return best;
  }
  // Intervals [root, third, fifth, seventh] stacked in thirds *within the scale*,
  // so every chord is built from the key's own notes (Am in C is A–C–E–G, not A–C#–E–G#).
  function chordIntervals(off){
    if(chromaticScale){
      const maj = (scaleName==='major'||scaleName==='mixolydian'||scaleName==='lydian'||scaleName==='major_pentatonic');
      return [0, maj?4:3, 7, 10];
    }
    const L = scalePcs.length, i = scalePcs.indexOf(off);
    const at = k => { const j = i + k; return scalePcs[j % L] + 12*Math.floor(j / L) - off; };
    let third = at(2), fifth = at(4), seventh = at(6);
    if(L !== 7){
      const has = iv => scalePcs.includes((off + iv) % 12);
      const t = [4,3,5,2].find(has); if(t!=null) third = t;
      const f = [7,8,6,9,10].find(iv => has(iv) && Math.abs(iv-third)!==1); if(f!=null) fifth = f;
      const sv = [10,11,9,12].find(iv => has(iv%12) && Math.abs(iv-fifth)!==1); seventh = (sv!=null) ? sv : 12;
    }
    // keep tones stacked upward inside an octave-and-a-bit
    while(fifth <= third) fifth += 12;
    while(seventh <= fifth) seventh += 12;
    if(seventh > 14) seventh -= 12;
    if(seventh <= fifth) seventh = fifth + 3;
    return [0, third, fifth, seventh];
  }
  // chord progression — only use progressions whose roots belong to the scale
  const progFits = arr => chromaticScale || arr.every(d => scalePcs.includes(((d%12)+12)%12));
  let prog = null;
  if(ch.chord_progression && CHORD_PROGRESSIONS[ch.chord_progression]){
    prog = CHORD_PROGRESSIONS[ch.chord_progression];
  } else if(ch.chord_progression === null){
    prog = null; // free / atonal
  } else {
    const keys = Object.keys(CHORD_PROGRESSIONS);
    const fitting = keys.filter(k => progFits(CHORD_PROGRESSIONS[k]));
    const pool = fitting.length ? fitting : keys;
    prog = CHORD_PROGRESSIONS[pool[Math.floor(rng()*pool.length)]];
  }
  // resolve chord roots to midi (octave 3), snapped into the scale
  const chordOffsets = prog ? prog.map(d => snapToScale(d)) : null;
  const chordRoots = chordOffsets ? chordOffsets.map(o => rootMidi + o) : null;
  const chordIvs = chordOffsets ? chordOffsets.map(o => chordIntervals(o)) : null;

  const timeline = [];
  const sections = []; // {name, start, end}
  let t = 0;

  let _chordEnd = Infinity;   // notes may not ring past the current chord change
  let _currentGridStep = 0;
  let _currentBaseTime = 0;
  let _currentRole = 'lead';
  let _sectionStartStep = 0;
  function addNote(voice, midi, time, dur, params){
    if(midi==null) return;
    if(time < _chordEnd && time + dur > _chordEnd) dur = Math.max(0.05, _chordEnd - time - 0.02);
    // soft ceiling: keep melodic material below ~E6 to avoid pitchy tops
    let m = midi;
    while(m > 88) m -= 12; // E6
    const freq = midiToFreq(m);
    const _stepOff = Math.max(0, Math.round((time - _currentBaseTime) / stepDur));
    timeline.push({kind:'note', voice, freq, midi:m, time, dur, gridStep:_currentGridStep + _stepOff, role:_currentRole, params:params||{}});
  }
  function addDrum(drum, time, params){
    // Route generic "kick" to genre-appropriate body
    if(drum==='kick'){
      const id = ch.id||'';
      const cl = ch.cluster||'';
      if(cl==='rock' || /metal|punk|grunge|rock|doom/.test(id)) drum = 'kick_acoustic';
      else if(cl==='electronic' || /house|techno|trance|electro|garage/.test(id)) drum = 'kick_909';
      else if(cl==='hiphop' || /trap|boom|lofi|drill|phonk/.test(id)) drum = 'kick_808';
      else if(cl==='jazz' || cl==='classical' || cl==='folk') drum = 'kick_acoustic';
    }
    if(drum==='snare' && (ch.cluster==='electronic' || /house|techno|trance/.test(ch.id||''))){
      drum = 'snare_909';
    }
    const _dStepOff = Math.max(0, Math.round((time - _currentBaseTime) / stepDur));
    timeline.push({kind:'drum', drum, time, gridStep:_currentGridStep + _dStepOff, params:params||{}});
  }
  function addTexture(name, time, dur, params){
    timeline.push({kind:'texture', name, time, dur, params:params||{}});
  }
  function addSfx(name, time, params){
    timeline.push({kind:'sfx', name, time, params:params||{}});
  }

  // pick a melody range — keep mostly octave 4 so leads aren't pitchy
  const melBase = rootMidi + 12; // rootMidi is ~48 → mel around 60
  const bassBase = rootMidi - 12;

  // Voice-leading: keep previous voicing and choose inversion minimizing displacement
  let _prevVoicing = null;
  // Returns the chord voicing in ROLE order [root, third, fifth, seventh]
  // (so index 0 is always the root), plus .root (true root, octave 3) and .ivs.
  function chordNotes(chordIdx){
    if(!chordRoots){
      // free/atonal channels still get a stable triad on the key root
      const r = rootMidi;
      const out = [r, r+3, r+7, r+12]; out.root = r; out.ivs = [0,3,7,12];
      return out;
    }
    const k = chordIdx % chordRoots.length;
    const root = chordRoots[k];
    // Sevenths only where they're idiomatic; elsewhere the 4th role doubles the root an octave up.
    const sevenths = ch.cluster==='jazz' || /lofi|neo_soul|rnb|city|bossa|lounge|smooth|vapor|chill|future_funk|shibuya|deep_house|nu_jazz/.test(ch.id||'');
    const iv = sevenths ? chordIvs[k] : [chordIvs[k][0], chordIvs[k][1], chordIvs[k][2], 12];
    const r = 48 + (((root - 48) % 12) + 12) % 12;      // root in 48–59
    const candidates = [
      [r, r+iv[1], r+iv[2], r+iv[3]],                    // close
      [r, r+iv[1], r+iv[2]+12, r+iv[3]+12],              // spread
      [r+12, r+iv[1], r+iv[2], r+iv[3]],                 // 1st inversion
      [r+12, r+iv[1]+12, r+iv[2], r+iv[3]],              // 2nd inversion
      [r+12, r+iv[1]+12, r+iv[2]+12, r+iv[3]+12],        // up an octave
    ];
    function cost(v){
      if(!_prevVoicing) return Math.abs(v[0]-r);
      let c = 0;
      for(let i=0;i<4;i++){ const d = v[i]-_prevVoicing[i]; c += d*d; }
      return c;
    }
    candidates.sort((a,b)=> cost(a)-cost(b));
    const best = candidates[0].slice();
    _prevVoicing = best.slice();
    best.root = r; best.ivs = iv;
    return best;
  }

  // section parameter overrides
  function sectionParams(name){
    const dyn = (ch.dynamics && ch.dynamics[name]!=null) ? ch.dynamics[name] : 0.7;
    let dens = ch.density;
    if(name==='chorus') dens *= 1.15;
    else if(name==='prechorus') dens *= 1.08;
    else if(name==='bridge') dens *= 0.55;
    else if(name==='intro') dens *= 0.55;
    else if(name==='outro') dens *= 0.6;
    else if(name==='verse') dens *= 1.0;
    return { dyn, dens };
  }

  function swingOffset(step){
    if(swing<=0) return 0;
    return (step%2===1) ? swing*stepDur : 0;
  }

  // ── SECTION GENERATOR ──
  // Genre-aware generation: motifs, walking/pulse bass, 7th chords, comping,
  // phrase structure, chord-tone targeting on strong beats.
  function genSection(name, startTime, sectionIdx){
    const bars = (ch.sectionBars && ch.sectionBars[name]) || 8;
    const totalSteps = bars * 16;  // 16 sixteenth-note steps per bar
    const sp = sectionParams(name);
    const isBeatless = ch.beatless;
    const patternName = (ch.drums && ch.drums.pattern) || 'four_floor';
    const pat = DRUM_PATTERNS[patternName] || DRUM_PATTERNS.four_floor;
    const drumCfg = ch.drums || {};
    const voices = ch.voices || ['piano_chop'];
    const leadParams = ch.leadVoiceParams || {};
    const roles = buildVoiceRoles(ch, rng);
    const style = getChannelStyle(ch);
    let leadVoice = roles.leadFor(name);
    let bassVoice = roles.bass;
    let padVoice = roles.pad;
    let compVoice = roles.comp;
    let arpVoice = roles.arp;
    let fillVoice = roles.fill;
    let secondLead = roles.secondaryLead;
    // Channel style locks (genre fingerprint)
    if(style){
      if(style.lockLead && VOICES[style.lockLead]) leadVoice = style.lockLead;
      if(style.lockBass && VOICES[style.lockBass]) bassVoice = style.lockBass;
      if(style.lockPad && VOICES[style.lockPad]) padVoice = style.lockPad;
      if(style.pad==='none') padVoice = null;
    }
    // Extra color voices from the full pool so more of the 9 get heard
    const colorPool = roles.pool.filter(v => v!==leadVoice && v!==bassVoice && v!==padVoice && isLeadish(v));
    const colorA = colorPool[0] || secondLead;
    const colorB = colorPool[1] || fillVoice || secondLead;
    const cluster = ch.cluster;
    const isJazz = cluster==='jazz' || scaleName==='dorian' || scaleName==='mixolydian';
    const isElectronic = cluster==='electronic';
    const isHiphop = cluster==='hiphop';
    const isRock = cluster==='rock';
    const isFolk = cluster==='folk';
    const isClassical = cluster==='classical';
    const isWorld = cluster==='world';
    const isAmbient = cluster==='ambient';
    const isRetro = cluster==='retro';
    const isPop = cluster==='pop';
    const isJapanese = cluster==='japanese';

    // chord changes: one chord per bar (or per 2 bars for slower tempos)
    const chordPerBars = bpm < 90 ? 2 : 1;
    const stepsPerBar = 16;

    // intro: sparse, no full kit
    const drumsActive = !isBeatless && !(name==='intro' && chance(rng,0.5));
    const drumsReduce = (name==='outro');

    // ── MOTIF GENERATION (chord-relative, grid-locked) ──
    // Motif stores chord-tone indices (0=root,1=3rd,2=5th,3=7th) + optional
    // scale-pass flag, and only lands on 8th-note (or sparser) grid.
    function makeMotif(){
      const motif = [];
      const phraseSteps = 2 * stepsPerBar;
      // Prefer simpler, more musical grids — avoid "busy" smash patterns
      const rhythmsSteady = [
        [0, 4, 8, 12, 16, 20, 24, 28],
        [0, 4, 8, 16, 20, 24],
        [0, 4, 12, 16, 20, 28],
      ];
      const rhythmsSparse = [
        [0, 8, 16, 24],
        [0, 8, 12, 16, 24, 28],
        [0, 6, 8, 16, 22, 24],
      ];
      const rhythmsBusy = [
        [0, 4, 8, 12, 16, 20, 24, 28],
        [0, 4, 6, 8, 12, 16, 20, 22, 24, 28],
        [0, 4, 8, 12, 14, 16, 20, 24, 28],
      ];
      const motifKind = (style && style.motif) || 'steady';
      const rhythms = motifKind==='sparse' ? rhythmsSparse : (motifKind==='busy' ? rhythmsBusy : rhythmsSteady);
      const rhythm = pick(rng, rhythms);
      // Contours as chord-tone index walks (0–3), not random scale degrees
      const contours = [
        [0, 1, 2, 1, 0, 2, 1, 0],
        [0, 2, 3, 2, 0, 1, 0, 2],
        [2, 1, 0, 1, 2, 3, 2, 0],
        [0, 1, 0, 2, 0, 1, 2, 0],
        [1, 2, 1, 0, 1, 2, 3, 2],
        [0, 2, 0, 2, 1, 0, 2, 0],
        [0, 1, 3, 2, 1, 0, 2, 3],           // leap to 7th then settle
        [2, 0, 1, 0, 2, 1, 3, 1],           // 5th-root bounce
        [0, 0, 1, 2, 2, 3, 2, 1],           // repeated then climb
        [3, 2, 1, 0, 1, 0, 2, 0],           // high start descend
        [1, 0, 1, 2, 0, 1, 0, 2],           // 3rd-root alternation
        [0, 2, 1, 3, 2, 0, 1, 2],           // open intervals
      ];
      const contour = pick(rng, contours);
      for(let i=0;i<rhythm.length;i++){
        const step = rhythm[i];
        const ctIdx = ((contour[i % contour.length] % 4) + 4) % 4;
        const nextStep = (i+1 < rhythm.length) ? rhythm[i+1] : phraseSteps;
        const durSteps = Math.max(2, nextStep - step); // at least an 8th
        // occasional scale approach (will resolve to chord tone)
        const approach = (i>0 && chance(rng,0.22));
        motif.push({ ctIdx, step, durSteps, approach });
      }
      return motif;
    }

    const motifA = makeMotif();
    let motifB = makeMotif();
    // Ensure B phrase differs from A (reduce loop feel)
    let tries=0;
    while(tries++<6 && motifB.length===motifA.length && motifB.every((m,i)=>motifA[i]&&m.ctIdx===motifA[i].ctIdx&&m.step===motifA[i].step)){
      motifB = makeMotif();
    }
    // Extra motifs for later sections so verse2/chorus2 aren't clones
    const motifC = makeMotif();
    const motifD = makeMotif();
    let useArpeggio = isElectronic || isRetro || isJapanese || isPop || (leadVoice&&leadVoice.includes('arp')) || (leadVoice&&leadVoice.includes('chiptune')) || ch.id==='synthwave';
    if(style){
      if(style.arp===true) useArpeggio = true;
      if(style.arp===false) useArpeggio = false;
    }

    // Map chord-tone index → MIDI from current chordNotes, with safe octave
    function chordToneMidi(cnotes, ctIdx, octShift){
      if(!cnotes || !cnotes.length) return melBase;
      const n = cnotes[ctIdx % cnotes.length];
      // Default: keep melody in upper-mid register (above pads)
      const base = n + ((octShift!=null)?octShift:1)*12;
      // Soft clamp into a singable band (~G3–E5)
      let m = base;
      while(m < 55) m += 12;
      while(m > 76) m -= 12;
      return m;
    }
    // Diatonic neighbor: ±1 scale step from a MIDI pitch (no chromatic clashes)
    function diatonicNeighbor(midi, dir){
      // find closest scale note, then step along scaleArr
      let idx = 0, bestD = 999;
      // scaleArr spans ~3 octaves — search it directly
      for(let i=0;i<scaleArr.length;i++){
        const d = Math.abs(scaleArr[i] - midi);
        if(d < bestD){ bestD = d; idx = i; }
      }
      const ni = Math.max(0, Math.min(scaleArr.length-1, idx + (dir>=0?1:-1)));
      return scaleArr[ni];
    }


    // ── main step loop ──
    for(let step=0; step<totalSteps; step++){
      const tt = startTime + step*stepDur + swingOffset(step);
      _currentGridStep = _sectionStartStep + step;
      _currentBaseTime = tt;
      const barIdx = Math.floor(step / stepsPerBar);
      const sInBar = step % stepsPerBar;
      const chIdx2 = Math.floor(barIdx / chordPerBars) % (chordRoots ? chordRoots.length : 1);
      const cnotes = chordNotes(chIdx2);
      const chordRoot = cnotes.root;
      const chordStart = (barIdx % chordPerBars) === 0;
      _chordEnd = startTime + (Math.floor(barIdx/chordPerBars)+1)*chordPerBars*stepsPerBar*stepDur + 0.001;
      const I3 = cnotes.ivs[1], I5 = cnotes.ivs[2], I7 = cnotes.ivs[3];

      // ── drums ──
      if(drumsActive){
        if(drumCfg.kick && pat.kick[sInBar]) addDrum('kick', tt);
        if(drumCfg.snare && pat.snare[sInBar] && !drumsReduce) addDrum('snare', tt);
        if(drumCfg.clap && pat.clap[sInBar] && !drumsReduce) addDrum('clap', tt);
        if(drumCfg.hihat && pat.hatc[sInBar]) addDrum('hatc', tt, {vel:0.28});
        if(drumCfg.hihat && pat.hato[sInBar]) addDrum('hato', tt, {vel:0.24});
        if(drumCfg.rimshot && pat.rim && pat.rim[sInBar]) addDrum('rim', tt);
        if(drumCfg.snap && pat.snap && pat.snap[sInBar]) addDrum('snap', tt);
        if(drumCfg.cowbell && pat.cowbell && pat.cowbell[sInBar]) addDrum('cowbell', tt);
        if(drumCfg.clave && pat.clave && pat.clave[sInBar]) addDrum('clave', tt);
        if(drumCfg.ride && pat.ride && pat.ride[sInBar] && (sInBar%4===0 || sInBar%8===6)) addDrum('ride', tt, {vel:0.12});
        if(drumCfg.crash && pat.crash && pat.crash[sInBar] && sInBar===0 && (ch.cluster==='rock' || /metal|punk|grunge/.test(ch.id||''))) addDrum('crash', tt, {vel:0.14});
        if(drumCfg.shaker && pat.shaker && pat.shaker[sInBar]) addDrum('shaker', tt, {vel:0.25});
        if(drumCfg.djembe && pat.perc && pat.perc[sInBar]) addDrum('djembe', tt);
        if(drumCfg.tom && pat.tom && pat.tom[sInBar]) addDrum('tom', tt);
        // trap hi-hat rolls
        if(patternName==='trap' && sInBar%4===2 && chance(rng,0.18)){
          addDrum('hatc', tt+stepDur*0.5, {vel:0.18});
        }
        // Euclidean overlays — genre syncopation (tresillo / cinquillo / bossa / afro)
        if(typeof EUCLID !== 'undefined' && sInBar < 16){
          const id = ch.id||'';
          if(/reggae|dub|dancehall|cumbia|lofi/.test(id) && EUCLID.tresillo){
            // map 8-step tresillo onto 16ths: each euclid step = 2 sixteenths
            const e = EUCLID.tresillo();
            if(sInBar%2===0 && e[sInBar/2] && !pat.kick[sInBar] && drumCfg.kick && chance(rng,0.35)){
              addDrum('rim', tt, {vel:0.3});
            }
          }
          if(/afrobeat|samba|uk_garage|salsa/.test(id)){
            const e = EUCLID.cinquillo();
            if(sInBar%2===0 && e[sInBar/2] && drumCfg.hihat && chance(rng,0.4)){
              addDrum('hatc', tt, {vel:0.28});
            }
          }
          if(/bossa|nu_jazz|smooth_jazz|lounge|shibuya/.test(id)){
            const e = EUCLID.bossa();
            if(e[sInBar] && drumCfg.rimshot && chance(rng,0.5)){
              addDrum('rim', tt, {vel:0.32});
            }
          }
          if(/african|afrobeat|samba|funk/.test(id)){
            const e = EUCLID.afro();
            if(e[sInBar] && drumCfg.shaker && chance(rng,0.45)){
              addDrum('shaker', tt, {vel:0.25});
            }
          }
        }
      }

      // ── BASS (channel-style first, then cluster fallback) ──
      _currentRole = 'bass';
      if(!isBeatless && !(style && style.bass==='none')){
        const bm = chordRoot - 12;
        const bmode = style && style.bass;
        if(bmode==='lofi'){
          // soft root on 1, occasional 5th, long notes, never busy
          if(sInBar===0) addNote(bassVoice, bm, tt, spb*3.2, {vel: sp.dyn*0.85});
          else if(sInBar===8 && chance(rng,0.3)) addNote(bassVoice, bm+I5, tt, spb*1.8, {vel: sp.dyn*0.55});
        } else if(bmode==='trap'){
          // long 808 on downbeat only, sparse
          if(sInBar===0) addNote(bassVoice, bm, tt, spb*3.8, {vel: sp.dyn*0.98});
          else if(sInBar===8 && chance(rng,0.2)) addNote(bassVoice, bm-12, tt, spb*2, {vel: sp.dyn*0.7});
        } else if(bmode==='boom'){
          if(sInBar===0) addNote(bassVoice, bm, tt, spb*2.4, {vel: sp.dyn*0.95});
          else if(sInBar===8) addNote(bassVoice, bm+(chance(rng,0.5)?0:I5), tt, spb*1.6, {vel: sp.dyn*0.7});
          else if(sInBar===12 && chance(rng,0.35)) addNote(bassVoice, bm, tt, spb, {vel: sp.dyn*0.6});
        } else if(bmode==='house' || bmode==='disco'){
          // four-on-the-floor adjacent: root every beat
          if(sInBar%4===0) addNote(bassVoice, bm, tt, spb*0.95, {vel: sp.dyn*0.88});
        } else if(bmode==='pulse'){
          if(sInBar%4===0) addNote(bassVoice, (sInBar===0)?bm:bm+(chance(rng,0.2)?12:0), tt, spb*0.9, {vel: sp.dyn*0.82});
        } else if(bmode==='techno'){
          if(sInBar%4===0) addNote(bassVoice, bm, tt, spb*0.85, {vel: sp.dyn*0.9});
          else if(sInBar%4===2 && chance(rng,0.4)) addNote(bassVoice, bm+12, tt, stepDur*2, {vel: sp.dyn*0.5});
        } else if(bmode==='acid'){
          // 16th acid line on root/5th/octave
          if(sInBar%2===0) addNote(bassVoice, bm+pick(rng,[0,0,0,I5,12,-12]), tt, stepDur*1.6, {vel: sp.dyn*0.75});
        } else if(bmode==='dnb'){
          if(sInBar===0 || sInBar===6 || sInBar===10) addNote(bassVoice, bm, tt, spb*0.7, {vel: sp.dyn*0.9});
          else if(sInBar===4 && chance(rng,0.5)) addNote(bassVoice, bm+I5, tt, spb*0.5, {vel: sp.dyn*0.65});
        } else if(bmode==='slow'){
          if(sInBar===0 && chordStart) addNote(bassVoice, bm, tt, spb*chordPerBars*3.5, {vel: sp.dyn*0.75});
        } else if(bmode==='bossa'){
          if(sInBar===0) addNote(bassVoice, bm, tt, spb*1.2, {vel: sp.dyn*0.85});
          else if(sInBar===6) addNote(bassVoice, bm+I5, tt, spb*0.8, {vel: sp.dyn*0.65});
          else if(sInBar===10) addNote(bassVoice, bm+I3, tt, spb*0.7, {vel: sp.dyn*0.6});
          else if(sInBar===14) addNote(bassVoice, bm+I5, tt, stepDur*2, {vel: sp.dyn*0.55});
        } else if(bmode==='funk'){
          if(sInBar===0) addNote(bassVoice, bm, tt, stepDur*3, {vel: sp.dyn*0.9});
          else if(sInBar===4) addNote(bassVoice, bm, tt, stepDur*2, {vel: sp.dyn*0.7});
          else if(sInBar===6 && chance(rng,0.6)) addNote(bassVoice, bm+I5, tt, stepDur*2, {vel: sp.dyn*0.65});
          else if(sInBar===10) addNote(bassVoice, bm+12, tt, stepDur*2, {vel: sp.dyn*0.6});
          else if(sInBar===12) addNote(bassVoice, bm, tt, stepDur*3, {vel: sp.dyn*0.75});
        } else if(bmode==='city'){
          if(sInBar%4===0) addNote(bassVoice, bm+(sInBar===8&&chance(rng,0.3)?I5:0), tt, spb*0.9, {vel: sp.dyn*0.85});
        } else if(bmode==='chip'){
          if(sInBar%4===0) addNote(bassVoice, bm, tt, spb*0.85, {vel: sp.dyn*0.8});
          else if(sInBar%4===2 && chance(rng,0.5)) addNote(bassVoice, bm+12, tt, stepDur*2, {vel: sp.dyn*0.55});
        } else if(isJazz){
          // Full walking bass: chord tones + chromatic approach on beat 4
          if(sInBar % 4 === 0){
            const beat = Math.floor(sInBar / 4);
            let walkNote;
            if(beat === 0) walkNote = bm;
            else if(beat === 1) walkNote = bm + pick(rng, [I3, I5, I5-12]);
            else if(beat === 2) walkNote = bm + pick(rng, [I5, I7, 12, I3+12]);
            else {
              const nextRoot = (chordRoots && chordRoots.length)
                ? (chordRoots[(chIdx2+1) % chordRoots.length] - 12)
                : bm;
              walkNote = nextRoot + (nextRoot >= bm ? -1 : 1);
            }
            addNote(bassVoice, walkNote, tt, spb*0.92, {vel: sp.dyn*(beat===0?0.9:0.75)});
          }
          if(sInBar%4===2 && chance(rng,0.25)){
            addNote(bassVoice, bm+I5, tt, stepDur*1.5, {vel: sp.dyn*0.35});
          }
        } else if(isHiphop || patternName==='trap' || patternName==='boom_bap' || patternName==='lofi'){
          // Channel-level: trap = sparse long 808; lofi/boom_bap = more movement
          const isTrap = patternName==='trap' || /trap/.test(ch.id||'');
          const isLofi = patternName==='lofi' || /lofi/.test(ch.id||'');
          if(sInBar === 0){
            addNote(bassVoice, bm, tt, isTrap ? spb*3.5 : spb*2.2, {vel: sp.dyn*0.95});
          } else if(!isTrap && sInBar === 8 && chance(rng, isLofi?0.35:0.55)){
            addNote(bassVoice, bm + (chance(rng,0.5)?I5:12), tt, spb*1.4, {vel: sp.dyn*0.65});
          } else if(!isTrap && sInBar === 12 && chance(rng, 0.3)){
            addNote(bassVoice, bm, tt, spb*1.1, {vel: sp.dyn*0.65});
          }
        } else if(isElectronic || isRetro || isJapanese || isPop){
          // pulse bass: quarter notes on root (8ths only in chorus), stays harmonic
          const pulseGrid = (name==='chorus') ? 2 : 4;
          if(sInBar % pulseGrid === 0){
            const pulseNote = (sInBar === 0) ? bm : (chance(rng,0.25) ? bm+12 : bm);
            addNote(bassVoice, pulseNote, tt, stepDur*pulseGrid*0.9, {vel: sp.dyn*0.78});
          }
        } else if(isRock || isFolk){
          // root-fifth bass: root on 1 and 3, fifth on 2 and 4 (or root-fifth-octave)
          if(sInBar % 4 === 0){
            const beatNum = Math.floor(sInBar / 4);
            const rockNote = (beatNum % 2 === 0) ? bm : bm + I5;
            addNote(bassVoice, rockNote, tt, spb*0.9, {vel: sp.dyn*0.9});
          }
        } else if(isWorld){
          // world: root on downbeat + syncopated percussion-like bass
          if(sInBar === 0){
            addNote(bassVoice, bm, tt, spb*1.5, {vel: sp.dyn*0.9});
          } else if(sInBar % 4 === 0 && chance(rng, 0.4)){
            addNote(bassVoice, bm + pick(rng,[0,I5,12]), tt, spb*0.8, {vel: sp.dyn*0.6});
          }
        } else {
          // generic: root on chord change
          if(sInBar === 0){
            addNote(bassVoice, bm, tt, spb*2, {vel: sp.dyn*0.85});
          } else if(sInBar === 8 && chance(rng, 0.4)){
            addNote(bassVoice, bm, tt, spb*1.5, {vel: sp.dyn*0.7});
          }
        }
      }

      // ── CHORD PAD / COMP — thin voicings, low velocity, chord tones only ──
      _currentRole = 'chord';
      if(padVoice || compVoice){
        const third = chordRoot + I3;
        const fifth = chordRoot + I5;
        // Prefer triad (root+5th or root+3rd+5th) — avoid dense 7th stacks fighting the lead
        if(isJazz && compVoice){
          if((sInBar===4 || sInBar===12) && chance(rng, name==='chorus'?0.55:0.4)){
            // two-note stab (3rd+7th or root+3rd) — classic jazz comp space
            const a = third, b = chordRoot + I7;
            addNote(compVoice, a, tt, stepDur*2.5, {vel: sp.dyn*0.26});
            addNote(compVoice, b, tt, stepDur*2.5, {vel: sp.dyn*0.24});
          }
        } else if(isElectronic || isRetro || isJapanese || isPop){
          if(sInBar === 0 && chordStart && padVoice){
            const padChance = (style && style.pad==='soft') ? 0.55 : 1;
            if(chance(rng, padChance)){
              addNote(padVoice, chordRoot, tt, spb*chordPerBars*4*0.95, {vel: sp.dyn*0.26});
              addNote(padVoice, fifth, tt, spb*chordPerBars*4*0.95, {vel: sp.dyn*0.22});
              if((style && style.pad==='hold') || (name==='chorus' && chance(rng,0.5))){
                addNote(padVoice, third, tt, spb*chordPerBars*4*0.95, {vel: sp.dyn*0.18});
              }
            }
          }
        } else if(isAmbient || isClassical){
          if(sInBar === 0 && chordStart && padVoice){
            addNote(padVoice, chordRoot, tt, spb*chordPerBars*4*0.95, {vel: sp.dyn*0.3});
            addNote(padVoice, fifth, tt, spb*chordPerBars*4*0.95, {vel: sp.dyn*0.26});
            addNote(padVoice, third+12, tt, spb*chordPerBars*4*0.95, {vel: sp.dyn*0.2});
          }
        } else if(isRock){
          // Rock power chords on guitar (root+5th, +octave)
          const gv = (leadVoice==='guitar_distorted'||leadVoice==='guitar_clean') ? leadVoice
            : (roles.pool.find(v=>/guitar/.test(v)) || 'guitar_distorted');
          if(sInBar===0 && chordStart && VOICES[gv]){
            addNote(gv, chordRoot, tt, spb*chordPerBars*3.5, {vel: sp.dyn*0.55});
            addNote(gv, chordRoot+I5, tt, spb*chordPerBars*3.5, {vel: sp.dyn*0.5});
            addNote(gv, chordRoot+12, tt, spb*chordPerBars*3.5, {vel: sp.dyn*0.4});
          }
        } else {
          // generic: triad on downbeat only, quieter
          if(sInBar === 0 && chance(rng, 0.65)){
            const cv = compVoice || padVoice;
            addNote(cv, chordRoot, tt, spb*2.2, {vel: sp.dyn*0.3});
            addNote(cv, third, tt, spb*2.2, {vel: sp.dyn*0.26});
            addNote(cv, fifth, tt, spb*2.2, {vel: sp.dyn*0.24});
          }
        }
      }

      // ── LEAD MELODY (chord-tone motifs, grid-locked) ──
      _currentRole = 'lead';
      const phraseIdx = Math.floor(step / (2 * stepsPerBar));
      const stepInPhrase = step % (2 * stepsPerBar);
      // AABA: invert B by reversing chord-tone contour
      function invertMotif(m){
        if(!m || !m.length) return m;
        return m.map(n => ({ctIdx: (3 - (n.ctIdx||0)), step: n.step, durSteps: n.durSteps, approach: n.approach}));
      }
      const motifBin = invertMotif(motifB);
      // Section-aware phrase pools so later sections don't clone the intro
      let phrasePattern;
      if(name==='chorus' || name==='drop'){
        phrasePattern = [motifC, motifA, motifBin, motifC];
      } else if(name==='bridge' || name==='solo'){
        phrasePattern = [motifD, motifBin, motifC, motifD];
      } else if(name==='verse' && sectionIdx>=2){
        phrasePattern = [motifC, motifA, motifBin, motifC];
      } else {
        phrasePattern = [motifA, motifA, motifBin, motifB];
      }
      const motif = phrasePattern[phraseIdx % phrasePattern.length];

      let motifNote = null;
      for(const mn of motif){
        if(mn.step === stepInPhrase){ motifNote = mn; break; }
      }

      // Only play on actual motif hits — high reliability, not random chance smashing
      // Channel-aware melody density
      let playGate = name==='intro' ? 0.7 : (name==='outro' ? 0.75 : (name==='bridge' ? 0.8 : 0.92));
      if(style && style.melGate!=null){
        playGate = style.melGate * (name==='chorus'?1.08:name==='intro'?0.85:1);
      } else {
        if(/lofi|ambient|downtempo|vapor/.test(ch.id||'')) playGate *= 0.75;
        if(/trap|drill/.test(ch.id||'')) playGate *= 0.7;
        if(/jpop|anime|denpa|vocaloid|happy/.test(ch.id||'')) playGate = Math.min(0.98, playGate*1.05);
      }
      if(motifNote && chance(rng, playGate)){
        let midi = chordToneMidi(cnotes, motifNote.ctIdx, 0);
        // Approach tone: step from neighbor scale note into chord tone (short)
        if(motifNote.approach){
          if(isJazz && chance(rng, 0.55)){
            // Bebop chromatic enclosure: approach target from above then below (or reverse)
            const above = midi + 1;
            const below = midi - 1;
            if(chance(rng,0.5)){
              addNote(leadVoice, above, tt, stepDur*0.45, Object.assign({vel: sp.dyn*0.38}, leadParams));
              addNote(leadVoice, below, tt+stepDur*0.45, stepDur*0.45, Object.assign({vel: sp.dyn*0.38}, leadParams));
            } else {
              addNote(leadVoice, below, tt, stepDur*0.45, Object.assign({vel: sp.dyn*0.38}, leadParams));
              addNote(leadVoice, above, tt+stepDur*0.45, stepDur*0.45, Object.assign({vel: sp.dyn*0.38}, leadParams));
            }
            addNote(leadVoice, midi, tt+stepDur, Math.min(stepDur * Math.max(1, motifNote.durSteps-1), spb*2), Object.assign({vel: sp.dyn*0.72}, leadParams));
          } else {
            // diatonic neighbor
            const approachMidi = diatonicNeighbor(midi, chance(rng,0.5)?1:-1);
            addNote(leadVoice, approachMidi, tt, stepDur*0.85, Object.assign({vel: sp.dyn*0.4}, leadParams));
            midi = chordToneMidi(cnotes, motifNote.ctIdx, 0);
            addNote(leadVoice, midi, tt + stepDur, Math.min(stepDur * Math.max(1, motifNote.durSteps-1), spb*2), Object.assign({vel: sp.dyn*0.72}, leadParams));
          }
        } else {
          const dur = stepDur * motifNote.durSteps * 0.85;
          // Strong beats always root/3rd/5th; already chord tones
          addNote(leadVoice, midi, tt, Math.min(dur, spb*2.5), Object.assign({vel: sp.dyn*0.72}, leadParams));
        }
        // Harmonic counterline: only the chord 3rd or 5th above, never random intervals
        if(secondLead && secondLead !== leadVoice && chance(rng, (style&&style.secondRate!=null)?style.secondRate:0.22) && (sInBar===0 || sInBar===8)){
          const harmIdx = (motifNote.ctIdx + pick(rng, [1, 2])) % Math.max(3, cnotes.length);
          const harmMidi = chordToneMidi(cnotes, harmIdx, 0);
          addNote(secondLead, harmMidi, tt, stepDur * 2, {vel: sp.dyn*0.32});
        }
      }

      // ── ARPEGGIO — chord tones, mid density ──
      _currentRole = 'arp';
      if(useArpeggio && step%4===0 && chance(rng, Math.min(0.6, sp.dens*0.65))){
        const arpPattern = [0, 1, 2, 1];
        const arpIdx = Math.floor(step/4) % arpPattern.length;
        const arpNote = chordToneMidi(cnotes, arpPattern[arpIdx], 0);
        const av = (arpVoice && arpVoice !== leadVoice) ? arpVoice : (secondLead && secondLead !== leadVoice ? secondLead : null);
        if(av) addNote(av, arpNote, tt, stepDur*1.8, {vel: sp.dyn*0.3});
      }

      // ── obbligato / secondary texture (use more of the pool) ──
      _currentRole = 'color';
      if(secondLead && secondLead !== leadVoice && sInBar===4 && chance(rng, 0.5)){
        addNote(secondLead, chordToneMidi(cnotes, 1, 0), tt, spb*1.5, {vel: sp.dyn*0.28});
      }
      if(colorA && colorA !== leadVoice && sInBar===6 && chance(rng, 0.4)){
        addNote(colorA, chordToneMidi(cnotes, 2, 0), tt, stepDur*4, {vel: sp.dyn*0.26});
      }
      if(colorB && colorB !== leadVoice && colorB !== colorA && sInBar===10 && chance(rng, 0.4)){
        addNote(colorB, chordToneMidi(cnotes, 1, 0), tt, stepDur*3, {vel: sp.dyn*0.26}); // third, not root (root already in bass+pad)
      }
      if(fillVoice && fillVoice !== leadVoice && sInBar===14 && chance(rng, 0.35)){
        addNote(fillVoice, chordToneMidi(cnotes, pick(rng,[0,1,2]), 0), tt, stepDur*2, {vel: sp.dyn*0.26});
      }
      if(compVoice && !isJazz && sInBar===8 && chance(rng, 0.4)){
        addNote(compVoice, chordRoot, tt, spb, {vel: sp.dyn*0.22});
        addNote(compVoice, chordRoot+I5, tt, spb, {vel: sp.dyn*0.2});
      }

      // ── fill at bar end ──
      if(fillVoice && fillVoice !== leadVoice && sInBar===12 && chance(rng, 0.35 * sp.dens)){
        addNote(fillVoice, chordToneMidi(cnotes, pick(rng,[0,1,2]), 0), tt, stepDur*2, {vel: sp.dyn*0.28});
      }

      // ── bridge texture ──
      if(name==='bridge' && step===0){
        if(chance(rng,0.35)) addTexture('ambient_pad', tt, bars*spb*4, {vel:0.18, seed});
      }
    }

    _chordEnd = Infinity;
    // ── fills: end of phrase add a snare/hat fill ──
    if(drumsActive && !isBeatless && (name==='chorus' || name==='verse')){
      const fillStart = startTime + (bars-1)*spb*4;
      for(let s=12;s<16;s++){
        _currentGridStep = _sectionStartStep + (bars-1)*16 + s;
        _currentBaseTime = fillStart + s*stepDur;
        if(chance(rng,0.6)) addDrum(drumCfg.snare?'snare':'hatc', fillStart + s*stepDur, {vel:0.4});
      }
    }
    _currentGridStep = _sectionStartStep;
    _currentBaseTime = startTime;
    if(drumsActive && name==='intro' && bars>=4){
      addDrum('crash', startTime, {vel:0.3});
    }
    // crash at start of chorus for impact
    if(name==='chorus' && drumCfg.crash){
      addDrum('crash', startTime, {vel:0.35});
    }

    // ── NES / Genesis in-song SFX ──
    const chipy = isRetro || /chiptune|arcade|demoscene|denpa|vocaloid|anime_op|game/.test(ch.id||'');
    if(chipy){
      // Section openers
      if(name==='intro'){
        addSfx(chance(rng,0.5)?'powerup':'confirm', startTime+0.05, {vel:0.45});
        if(chance(rng,0.4)) addSfx('blip', startTime+0.2, {vel:0.3});
      }
      if(name==='chorus'){
        addSfx(chance(rng,0.4)?'levelclear':'coin', startTime+0.02, {vel:0.4});
      }
      if(name==='bridge'){
        addSfx(pick(rng,['teleport','laser','alarm']), startTime+0.05, {vel:0.4});
      }
      if(name==='outro'){
        addSfx(chance(rng,0.5)?'gameover':'fall', startTime + (bars*spb*4)*0.7, {vel:0.35});
      }
      // Occasional mid-section stingers on verse
      if(name==='verse' && chance(rng,0.35)){
        const mid = startTime + bars*spb*2;
        addSfx(pick(rng,['coin','jump','blip','select','ym_stab']), mid, {vel:0.35});
      }
      // Fill-end arcade blips
      if((name==='chorus'||name==='verse') && chance(rng,0.45)){
        const fillStart = startTime + (bars-1)*spb*4;
        addSfx(pick(rng,['blip','hit','coin']), fillStart + 12*stepDur, {vel:0.3});
      }
    }
    // Japanese anime OP gets powerup on prechorus→chorus
    if(isJapanese && name==='prechorus'){
      addSfx('powerup', startTime + bars*spb*3.5, {vel:0.4});
    }
    if(isJapanese && name==='chorus' && chance(rng,0.5)){
      addSfx(pick(rng,['coin','levelclear','confirm']), startTime, {vel:0.35});
    }

    // Never Mention Grok FM — mutation funk / phonk / vocaloid stingers
    if(ch.special==='never_mention_grok'){
      if(name==='intro'){
        addSfx('alarm', startTime+0.05, {vel:0.4});
        addSfx('blip', startTime+0.25, {vel:0.35});
      }
      if(name==='chorus'){
        addSfx(pick(rng,['powerup','levelclear','coin']), startTime, {vel:0.45});
        // chant energy: rapid blip train
        for(let b=0;b<4;b++){
          if(chance(rng,0.7)) addSfx('blip', startTime + b*spb + 0.05, {vel:0.28, freq:800+b*120});
        }
      }
      if(name==='bridge'){
        addSfx(pick(rng,['teleport','laser','static_burst']), startTime, {vel:0.45});
        addSfx('hit', startTime+spb, {vel:0.4});
      }
      if(name==='verse' && chance(rng,0.5)){
        addSfx(pick(rng,['coin','select','ym_stab','jump']), startTime + bars*spb*2, {vel:0.35});
      }
      if(name==='outro'){
        addSfx('gameover', startTime + bars*spb*2, {vel:0.4});
        addSfx('static_burst', startTime + bars*spb*3.2, {vel:0.35});
      }
    }

    // Guardrails FM — stomp/bluegrass intensity, sparse warning stingers
    if(ch.special==='guardrails_fm'){
      if(name==='intro'){
        addSfx('alarm', startTime+0.08, {vel:0.3});
      }
      if(name==='chorus'){
        addSfx(pick(rng,['hit','blip','select']), startTime+0.05, {vel:0.3});
      }
      if(name==='bridge'){
        addSfx('static_burst', startTime, {vel:0.35});
        addSfx('fall', startTime+spb*2, {vel:0.3});
      }
      if(name==='outro'){
        addSfx('gameover', startTime + bars*spb*2.5, {vel:0.35});
      }
    }

    // Baked In FM — phonk bounce, product-care energy, heavier SFX
    if(ch.special==='baked_in_fm'){
      if(name==='intro'){
        addSfx('motor', startTime, {vel:0.35, dur:0.5});
        addSfx('blip', startTime+0.3, {vel:0.3});
      }
      if(name==='chorus'){
        addSfx(pick(rng,['powerup','coin','confirm']), startTime, {vel:0.4});
        for(let b=0;b<3;b++){
          if(chance(rng,0.6)) addSfx('blip', startTime + b*spb*0.5, {vel:0.25, freq:900+b*100});
        }
      }
      if(name==='bridge'){
        addSfx(pick(rng,['teleport','laser','alarm']), startTime, {vel:0.4});
      }
      if(name==='verse' && chance(rng,0.4)){
        addSfx(pick(rng,['select','hit','ym_stab']), startTime + bars*spb*2, {vel:0.3});
      }
      if(name==='outro'){
        addSfx('static_burst', startTime + bars*spb*3, {vel:0.35});
      }
    }

    return startTime + bars*spb*4;
  }

  // ── build all sections ──
  const struct = ch.structure || ['intro','verse','chorus','verse','chorus','bridge','outro'];
  // pirate radio genre switch: swap descriptor halfway
  let activeCh = ch;
  for(let i=0;i<struct.length;i++){
    const sec = struct[i];
    if(ch.special==='genre_switch' && i===Math.floor(struct.length/2)){
      activeCh = getChannelForSong(CHANNELS[Math.floor(rng()*CHANNELS.length)], rng);
      // keep key & harmony, swap the band: genSection reads `ch`
      ch = Object.assign({}, activeCh, {chord_progression: ch.chord_progression, scale: ch.scale, structure: ch.structure, sectionBars: ch.sectionBars});
    }
    const secStart = t;
    const secStartStep = _sectionStartStep;
    const secBars = (ch.sectionBars && ch.sectionBars[sec]) || 8;
    t = genSection.call({
      // override channel refs if switched
    }, sec, t, i);
    _sectionStartStep += secBars * 16;
    sections.push({name:sec, start:secStart, end:t, startStep:secStartStep, endStep:_sectionStartStep, bars:secBars});
    // small gap between sections (humanise)
    t += 0;
  }

  // texture beds for ambient/beatless
  if(ch.beatless || ch.reverb>0.6){
    addTexture('ambient_pad', 0, t, {vel:0.09, seed});
  }
  if(ch.id==='lofi_hip_hop' || ch.cluster==='hiphop'){
    addTexture('vinyl_crackle', 0, t, {vel:0.045, seed});
  }
  if(ch.id==='dub' || ch.reverb>0.7){
    addTexture('tape_hiss', 0, t, {vel:0.04});
  }
  if(ch.special==='static'){
    addTexture('tape_hiss', 0, t, {vel:0.05});
  }
  if(ch.special==='never_mention_grok'){
    addTexture('tape_hiss', 0, t, {vel:0.03});
    addSfx('alarm', 0.1, {vel:0.35});
    addSfx('static_burst', 0.35, {vel:0.4});
  }
  if(ch.special==='guardrails_fm'){
    addTexture('tape_hiss', 0, t, {vel:0.025});
    addSfx('alarm', 0.15, {vel:0.28});
  }
  if(ch.special==='baked_in_fm'){
    addTexture('tape_hiss', 0, t, {vel:0.035});
    addSfx('motor', 0.1, {vel:0.3, dur:0.45});
    addSfx('static_burst', 0.4, {vel:0.35});
  }
  if(ch.special==='mostly_silence'){
    // only a couple of very sparse notes already generated by low density
  }

  // binaural: add a subtle beat-frequency drone
  _currentGridStep = 0; _currentBaseTime = 0; _currentRole = 'chord';
  if(ch.special==='binaural'){
    addNote('ambient_pad', rootMidi+12, 0, t, {vel:0.2});
    addNote('ambient_pad', rootMidi+12.5, 0, t, {vel:0.2}); // ~0.5Hz beating... (approx)
  }

  const duration = t;
  const info = {
    title: ch.name + ' #' + seed,
    bpm: Math.round(bpm),
    key: ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][key],
    scale: scaleName,
  };
  timeline.sort((a,b)=>a.time-b.time);
  return { timeline, duration, bpm, key, scaleName, sections, totalSteps: _sectionStartStep, info, channelName: ch.name, channelId: ch.id };
}

function pickBassVoice(voices){
  for(const v of voices){ if(/^bass_|reese|acid_bass/.test(v) && VOICES[v]) return v; }
  return 'bass_electric';
}

/* Role-based voice picker — uses the full channel palette instead of only leadVoice */
const CLUSTER_PALETTE = {
  electronic: ['supersaw','saw_pluck','synth_pad_warm','synth_pad_analog','bass_moog','acid_bass','synth_arp','fm_keys','fm_keys','synth_lead_saw','synth_lead_square','synth_pluck','wurlitzer','reese_bass','organ_hammond'],
  hiphop: ['piano_chop','bass_808','bass_electric','wurlitzer','guitar_muted','reese_bass','synth_pad_warm','fm_keys','piano_electric','clavinet','synth_bell','saw_pluck'],
  jazz: ['piano_acoustic','bass_acoustic','sax','vibraphone','guitar_clean','brass_section','flute_breathy','organ_hammond','pizz_strings','marimba','piano_electric','strings_section'],
  classical: ['piano_acoustic','strings_section','flute_breathy','choir_aaah','celesta','pizz_strings','brass_section','organ_pipe','music_box','ocarina','vibraphone','synth_pad_glass'],
  world: ['guitar_clean','bass_acoustic','flute_breathy','accordion','kalimba','sitar_drone','pan_flute','koto','dulcimer','ocarina','harmonica','banjo','marimba'],
  rock: ['guitar_distorted','bass_electric','organ_hammond','guitar_clean','synth_brass','piano_electric','guitar_muted','bass_moog','organ_gospel','synth_lead_saw'],
  ambient: ['synth_pad_warm','synth_pad_glass','synth_pad_analog','ambient_pad','kalimba','music_box','theremin','ocarina','celesta','flute_breathy','strings_section','synth_bell'],
  retro: ['ym2612_lead','ym2612_bell','ym2612_bass','chiptune_pulse','chiptune_triangle','chiptune_saw','fm_keys','bass_moog','synth_brass','wurlitzer','synth_bell','organ_hammond','saw_pluck'],
  folk: ['guitar_clean','banjo','harmonica','bass_acoustic','dulcimer','flute_breathy','accordion','guitar_muted','ocarina','kalimba','pizz_strings'],
  pop: ['supersaw','fm_keys','bass_electric','bass_moog','synth_pad_warm','guitar_clean','synth_bell','piano_electric','saw_pluck','synth_arp','wurlitzer','organ_hammond'],
  japanese: ['fm_keys','supersaw','synth_bell','bass_moog','chiptune_pulse','ym2612_lead','ym2612_bell','guitar_clean','wurlitzer','synth_pad_warm','saw_pluck','koto','synth_arp'],
  special: ['supersaw','bass_808','synth_bell','reese_bass','saw_pluck','fm_keys','chiptune_pulse','synth_pad_warm','ym2612_lead','synth_lead_saw','acid_bass','guitar_distorted'],
};

function isBassVoice(v){ return /^(bass_|reese|acid_bass)/.test(v); }
function isPadVoice(v){ return /pad|strings_section|choir|organ|ambient/.test(v); }
function isLeadish(v){ return VOICES[v] && !isBassVoice(v) && !/vinyl|tape|wind|rain|texture|cowbell|bell|vibraphone|marimba|kalimba|music_box|celesta|dulcimer|church|chime/.test(v); }

function buildVoiceRoles(ch, rng){
  let pool = (ch.voices || []).filter(v => VOICES[v]);
  // Always merge cluster palette so every song can draw from a wide set
  const extra = (CLUSTER_PALETTE[ch.cluster] || []).filter(v => VOICES[v] && !pool.includes(v));
  // Shuffle extras with rng, take enough to reach ~8–10 voices in the pool
  const shuffled = extra.slice();
  for(let i=shuffled.length-1;i>0;i--){
    const j = Math.floor(rng()*(i+1));
    const t = shuffled[i]; shuffled[i]=shuffled[j]; shuffled[j]=t;
  }
  const target = 9;
  for(const v of shuffled){
    if(pool.length >= target) break;
    pool.push(v);
  }
  // Seed-varied: sometimes swap in 1–2 more rare voices from the full registry
  const allNames = Object.keys(VOICES).filter(v => !pool.includes(v) && isLeadish(v));
  if(allNames.length && chance(rng, 0.45)){
    pool.push(allNames[Math.floor(rng()*allNames.length)]);
  }
  if(allNames.length > 1 && chance(rng, 0.25)){
    pool.push(allNames[Math.floor(rng()*allNames.length)]);
  }
  // always ensure a bass exists
  if(!pool.some(isBassVoice)){
    const fb = (ch.cluster==='hiphop'||ch.cluster==='special') ? 'bass_808'
      : (ch.cluster==='electronic'||ch.cluster==='retro'||ch.cluster==='japanese') ? 'bass_moog'
      : (ch.cluster==='jazz'||ch.cluster==='classical'||ch.cluster==='folk'||ch.cluster==='world') ? 'bass_acoustic'
      : 'bass_electric';
    if(VOICES[fb]) pool.push(fb);
  }
  const basses = pool.filter(isBassVoice);
  const pads = pool.filter(isPadVoice);
  const leads = pool.filter(v => isLeadish(v) && !isPadVoice(v));
  const keys = pool.filter(v => /piano|keys|wurlitzer|fm_keys|clavinet|harpsichord|celesta|vibraphone|marimba/.test(v));
  const plucks = pool.filter(v => /pluck|guitar|banjo|kalimba|koto|dulcimer|harp|pizz/.test(v));
  const chips = pool.filter(v => /chip|ym2612/.test(v));

  // Channel-spec voices get priority for leads (genre identity)
  const metallic = /dulcimer|music_box|marimba|kalimba|celesta|cowbell|ym2612_bell/;
  const channelLeads = (ch.voices||[]).filter(v => VOICES[v] && isLeadish(v) && !isBassVoice(v));
  // Prefer non-anvil leads unless the channel explicitly asks for them
  const softChannelLeads = channelLeads.filter(v => !metallic.test(v));
  const pickLead = softChannelLeads.length ? softChannelLeads : channelLeads;
  let primaryLead = (ch.leadVoice && VOICES[ch.leadVoice]) ? ch.leadVoice
    : (pickLead[0] || leads.filter(v=>!metallic.test(v))[0] || leads[0] || keys[0] || pool[0] || 'piano_acoustic');
  // If leadVoice is metallic but channel has softer options, keep it only when explicitly set
  const leadPool = (pickLead.length > 1 ? pickLead : leads.filter(v=>!metallic.test(v))).filter(v => v !== primaryLead);
  const secondaryLead = leadPool.length ? leadPool[Math.floor(rng()*leadPool.length)]
    : (keys.find(v=>v!==primaryLead) || plucks.filter(v=>!metallic.test(v))[0] || primaryLead);

  const bass = basses[Math.floor(rng()*basses.length)] || pickBassVoice(pool);
  const pad = pads.length ? pads[Math.floor(rng()*pads.length)]
    : (keys.find(v=>v!==primaryLead) || null);
  const comp = keys.length ? keys[Math.floor(rng()*keys.length)]
    : (plucks[0] || pad || primaryLead);
  const arp = pool.find(v => v.includes('arp')) || chips[0] || plucks[0] || secondaryLead;
  const fill = chips[0] || plucks[0] || keys[0] || secondaryLead;

  return {
    pool, primaryLead, secondaryLead, bass, pad, comp, arp, fill,
    // section-aware: chorus may flip to secondary
    leadFor(section){
      // Pick from lead pool so verses/choruses/bridges can sound different
      const opts = leads.length ? leads : [primaryLead];
      if(section==='intro'){
        if(pad && chance(rng,0.5)) return pad;
        return opts[Math.floor(rng()*Math.min(2, opts.length))];
      }
      if(section==='chorus'){
        // prefer secondary / alternate leads on chorus
        if(opts.length > 1) return opts[1 + Math.floor(rng()*Math.max(1, opts.length-1)) % opts.length];
        return secondaryLead || primaryLead;
      }
      if(section==='bridge' || section==='prechorus'){
        return secondaryLead || opts[Math.floor(rng()*opts.length)];
      }
      if(section==='outro'){
        return pad || primaryLead;
      }
      // verse: primary, occasional alternate
      if(opts.length > 1 && chance(rng,0.3)) return opts[Math.floor(rng()*opts.length)];
      return primaryLead;
    }
  };
}


/* ════════════════════════════════════════════════════════════════

/* ════════════════════════════════════════════════════════════════
   3. LIFTED AURALOOM SCORE PARSER & LOOM WEAVER (v0.4/v0.5)
   ════════════════════════════════════════════════════════════════ */

const AL_SR = 22050, AL_HOP = 256, AL_NR = 200, AL_F0 = 32.70319566, AL_DBR = 60;
const alRowOfFreq = f => 24 * Math.log2(f / AL_F0);
const alRowOfMidi = m => (m - 24) * 2;
const AURALOOM_SCALES = {
  chromatic:[0,1,2,3,4,5,6,7,8,9,10,11],
  major:[0,2,4,5,7,9,11],
  minor:[0,2,3,5,7,8,10],
  pentMaj:[0,2,4,7,9],
  pentMin:[0,3,5,7,10],
  dorian:[0,2,3,5,7,9,10],
  phrygian:[0,1,3,5,7,8,10],
  lydian:[0,2,4,6,7,9,11],
  mixolydian:[0,2,4,5,7,9,10],
  locrian:[0,1,3,5,6,8,10],
  harmonicMinor:[0,2,3,5,7,8,11],
  blues:[0,3,5,6,7,10],
  free:null
};
const AURALOOM_TIMBRES = {
  pure:   [[1,1]],
  flute:  [[1,1],[2,.35],[3,.12],[4,.05]],
  string: Array.from({length:12}, (_, i) => [i + 1, 1 / (i + 1)]),
  reed:   [[1,1],[2,.12],[3,.62],[4,.1],[5,.45],[6,.07],[7,.3],[9,.18],[11,.1]],
  organ:  [[1,1],[2,.8],[3,.55],[4,.5],[6,.3],[8,.3]],
  bell:   [[1,1],[2.76,.6],[5.4,.4],[8.93,.25],[13.34,.12]]
};
const AURALOOM_FM = {
  fmLead:  { R: 1,   i0: 2.4, i1: 1.5, tau: 0.30 },
  fmBass:  { R: 1,   i0: 3.4, i1: 0.7, tau: 0.08 },
  fmKeys:  { R: 2,   i0: 1.8, i1: 0.25, tau: 0.15 },
  fmBrass: { R: 1,   i0: 0.3, i1: 2.3, tau: 0.07 },
  fmBell:  { R: 3.5, i0: 3.0, i1: 0.4, tau: 0.6 },
  fmClang: { R: 1.41, i0: 2.6, i1: 0.9, tau: 0.25 }
};

function besselJ(n, x) {
  const m = Math.abs(n); let sum = 0, term = Math.pow(x / 2, m); for (let k = 1; k <= m; k++) term /= k;
  for (let k = 0; k < 30; k++) { sum += term; term *= -(x * x / 4) / ((k + 1) * (k + 1 + m)); if (Math.abs(term) < 1e-9) break; }
  return n < 0 && m % 2 ? -sum : sum;
}

const _partialsZeroCache = new Map();
function partialsAt(name, t) {
  const fm = AURALOOM_FM[name];
  if (!fm) return (AURALOOM_TIMBRES[name] || AURALOOM_TIMBRES.pure).map(([r, a], h) => [r, a, h]);
  if (!t && _partialsZeroCache.has(name)) return _partialsZeroCache.get(name);
  const I = fm.i1 + (fm.i0 - fm.i1) * Math.exp(-(t || 0) / fm.tau), N = Math.ceil(I) + 3, map = new Map();
  for (let n = -N; n <= N; n++) {
    const ratio = Math.abs(1 + n * fm.R); if (ratio < 0.02) continue;
    const key = ratio.toFixed(3), a = Math.abs(besselJ(n, I));
    map.set(key, [ratio, (map.get(key)?.[1] || 0) + a, Math.abs(n)]);
  }
  const list = [...map.values()].filter(p => p[1] > 0.004), top = Math.max(...list.map(p => p[1]));
  const res = list.map(p => [p[0], p[1] / top, p[2], true]);
  if (!t) _partialsZeroCache.set(name, res);
  return res;
}

const LOUD = l => 0.35 + 0.65 * Math.max(0, Math.min(100, l)) / 100;
function parseNote(tok) {
  const m = /^([A-Ga-g])([#♯b♭]?)(-?\d)(q?)$/.exec(tok); if (!m) return null;
  const pc = { c: 0, d: 2, e: 4, f: 5, g: 7, a: 9, b: 11 }[m[1].toLowerCase()] + (m[2] === '#' || m[2] === '♯' ? 1 : m[2] === 'b' || m[2] === '♭' ? -1 : 0);
  return (+m[3] + 1) * 12 + pc + (m[4] ? 0.5 : 0);
}
function parseWhere(tok, maxStep) {
  const out = [];
  for (const part of tok.split(',')) {
    let m;
    if ((m = /^(\d+)\.(\d+)(?:\.(\d+))?$/.exec(part))) out.push((+m[1] - 1) * 16 + (+m[2] - 1) * 4 + (m[3] ? +m[3] : 1));
    else if ((m = /^(\d+)-(\d+)(?:\/(\d+))?$/.exec(part))) { const a = +m[1], b = +m[2], k = m[3] ? +m[3] : 1; if (k < 1 || b < a) return null; for (let x = a; x <= b; x += k) out.push(x); }
    else if (/^\d+$/.test(part)) out.push(+part);
    else if ((m = /^(\d+(?:\.\d+)?)s$/.exec(part))) out.push(+m[1]);
    else return null;
  }
  return out.filter(x => x >= 1 && x <= maxStep);
}
const THREADS = ['tone', 'pluck', 'kick', 'snare', 'hat', 'openhat', 'noise', 'erase'];

/**
 * Exact headless port of AuraLoom's weaveScore(text) from auraloom/index.html lines 1914-2003.
 */
function weaveScore(text, renderGrid = false) {
  const lines = text.split(/\r?\n/), errs = [], set = {}, events = [], body = [];
  lines.forEach((raw, li) => {
    const line = raw.replace(/(^|\s)#.*/, '').trim(); if (!line) { body.push(raw); return; }
    const tk = line.split(/\s+/), head = tk[0].toLowerCase(), n = li + 1;
    if (head === 'bpm') { const v = +tk[1]; if (v >= 40 && v <= 240) set.bpm = Math.round(v); else errs.push(n + ': bpm must be 40-240'); return; }
    if (head === 'bars') { const v = +tk[1]; if ([1, 2, 4, 8].includes(v)) set.bars = v; else errs.push(n + ': bars must be 1, 2, 4 or 8'); return; }
    if (head === 'key') {
      const r = /^([A-Ga-g])([#♯b♭]?)$/.exec(tk[1] || '');
      const want = (tk.slice(2).join('') || 'major').toLowerCase().replace(/[^a-z]/g, '');
      const sc = Object.keys(AURALOOM_SCALES).find(k => k.toLowerCase() === want) || want;
      if (!r || !(sc in AURALOOM_SCALES)) { errs.push(n + ': key looks like "key A minor"'); return; }
      set.root = ((({ c: 0, d: 2, e: 4, f: 5, g: 7, a: 9, b: 11 }[r[1].toLowerCase()] + (r[2] === '#' || r[2] === '♯' ? 1 : r[2] ? -1 : 0)) + 12) % 12);
      set.scale = sc;
      return;
    }
    if (head === 'voice') { if (AURALOOM_TIMBRES[tk[1]] || AURALOOM_FM[tk[1]]) set.voice = tk[1]; else errs.push(n + ': unknown voice "' + tk[1] + '"'); return; }
    if (head === 'ring' || head === 'space' || head === 'crunch') { const v = +tk[1]; if (isFinite(v)) set[head] = v; else errs.push(n + ': ' + head + ' needs a number'); return; }
    if (head === 'mode') { if (tk[1] === 'clean' || tk[1] === 'ghost') set.mode = tk[1]; else errs.push(n + ': mode is clean or ghost'); return; }
    const thread = (tk[1] || '').toLowerCase();
    if (!THREADS.includes(thread)) { errs.push(n + ': expected "<where> <thread> ...", e.g. "5 snare"'); return; }
    const opts = {}, pos = [];
    for (const t of tk.slice(2)) { const kv = /^(\w+)=(.+)$/.exec(t); if (kv) opts[kv[1].toLowerCase()] = kv[2]; else pos.push(t); }
    events.push({ n, where: tk[0], thread, pos, opts }); body.push(raw);
  });
  const B = set.bars || 2, BP = set.bpm || 104;
  const defVoice = set.voice || 'string';
  const secs = B * 4 * 60 / BP;
  const cols = Math.round(secs * AL_SR / AL_HOP);
  const sc = cols / (secs * BP / 15);
  const colOf = st => Math.round((st - 1) * sc);
  let tone = renderGrid ? new Float32Array(AL_NR * cols) : null;
  let noise = renderGrid ? new Float32Array(AL_NR * cols) : null;
  const put = (L, r, c, v) => {
    r = Math.round(r); c = Math.round(c);
    if (r < 0 || r >= AL_NR || c < 0 || c >= cols || v <= 0) return;
    const i = r * cols + c; if (v > L[i]) L[i] = Math.min(1, v);
  };
  const rub = (r, c, rad) => {
    for (let dr = -rad; dr <= rad; dr++) for (let dc = -rad; dc <= rad; dc++) if (dr * dr + dc * dc <= rad * rad) {
      const rr = Math.round(r + dr), cc = Math.round(c + dc);
      if (rr >= 0 && rr < AL_NR && cc >= 0 && cc < cols) { tone[rr * cols + cc] = 0; noise[rr * cols + cc] = 0; }
    }
  };

  let woven = 0;
  let renderedEvents = 0;
  for (const ev of events) {
    const steps = parseWhere(ev.where, B * 16);
    if (!steps) { errs.push(ev.n + ': can\'t read the position "' + ev.where + '"'); continue; }
    if (!steps.length) { errs.push(ev.n + ': position is outside the ' + B + '-bar loop'); continue; }
    const voice = ev.opts.voice || defVoice;
    if (!AURALOOM_TIMBRES[voice] && !AURALOOM_FM[voice]) { errs.push(ev.n + ': unknown voice "' + voice + '"'); continue; }
    const nums = ev.pos.filter(t => /^\d+(\.\d+)?$/.test(t)).map(Number), words = ev.pos.filter(t => !/^\d+(\.\d+)?$/.test(t));
    const isPitched = ev.thread === 'tone' || ev.thread === 'pluck' || ev.thread === 'noise' || ev.thread === 'erase';
    let len = ev.opts.len != null ? +ev.opts.len : isPitched ? nums[0] : undefined;
    let loud = ev.opts.loud != null ? +ev.opts.loud : isPitched ? nums[1] : nums[0];
    const v = LOUD(loud != null && isFinite(loud) ? loud : 80);
    let notes = null, glide = null;
    if (isPitched && words.length) {
      const w = words[0];
      if (w.includes('>')) {
        if (ev.thread !== 'tone') { errs.push(ev.n + ': glide is only for tone'); continue; }
        const [a, b] = w.split('>').map(parseNote);
        if (a == null || b == null) { errs.push(ev.n + ': can\'t read the glide "' + w + '"'); continue; }
        glide = [a, b]; notes = [a];
      } else {
        notes = w.split('+').map(parseNote);
        if (notes.some(x => x == null)) { errs.push(ev.n + ': can\'t read the note "' + w + '" (try A4, C#5, Bb3)'); continue; }
      }
    }
    if ((ev.thread === 'tone' || ev.thread === 'pluck') && !notes) { errs.push(ev.n + ': ' + ev.thread + ' needs a note, e.g. "' + ev.where + ' ' + ev.thread + ' A4"'); continue; }
    for (const st of steps) {
      const c0 = colOf(st);
      if (c0 < 0 || c0 >= cols) { errs.push(ev.n + ': step ' + st + ' out of bounds'); continue; }
      if (renderGrid) {
        if (ev.thread === 'kick') {
          const n = Math.round(0.22 * AL_SR / AL_HOP);
          for (let dc = 0; dc < n; dc++) { const f = 46 + 120 * Math.exp(-dc / 2.2); put(tone, alRowOfFreq(f), c0 + dc, v - 0.5 * dc / n); }
        } else if (ev.thread === 'snare') {
          put(noise, Math.round(alRowOfFreq(400)), c0, v * 0.9);
        } else if (ev.thread === 'hat' || ev.thread === 'openhat') {
          put(noise, Math.round(alRowOfFreq(6000)), c0, v * 0.9);
        } else if (ev.thread === 'pluck') {
          for (const m of notes) put(tone, alRowOfMidi(m), c0, v);
        } else if (ev.thread === 'tone') {
          for (const m of notes) put(tone, alRowOfMidi(m), c0, v);
        } else if (ev.thread === 'noise') {
          const centre = notes ? alRowOfMidi(notes[0]) : alRowOfMidi(84);
          put(noise, centre, c0, v);
        } else if (ev.thread === 'erase') {
          const centre = notes ? alRowOfMidi(notes[0]) : alRowOfMidi(60);
          rub(centre, c0, ev.opts.width != null ? +ev.opts.width : 4);
        }
      }
      if (ev.thread !== 'erase') renderedEvents++;
      woven++;
    }
  }
  if (!errs.length && (woven === 0 || renderedEvents === 0)) errs.push('No notes found — check the format guide');
  return { woven, renderedEvents, errs, bpm: BP, bars: B, cols };
}

/* ════════════════════════════════════════════════════════════════
   4. SYNTH RADIO -> .LOOM SCORE EXPORTER
   ════════════════════════════════════════════════════════════════ */

const NOTE_NAMES_SHARP = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const NOTE_NAMES_FLAT  = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'];
const FLAT_SCALES = new Set(['minor', 'dorian', 'phrygian', 'locrian', 'pentMin', 'harmonicMinor', 'blues']);
const FLAT_ROOTS  = new Set([2, 3, 5, 7, 8, 10]); // D, Eb, F, G, Ab, Bb

function midiToLoomNote(midi, useFlats) {
  const m = Math.max(24, Math.min(108, Math.round(midi)));
  const pc = ((m % 12) + 12) % 12;
  const oct = Math.floor(m / 12) - 1;
  const names = useFlats ? NOTE_NAMES_FLAT : NOTE_NAMES_SHARP;
  return names[pc] + oct;
}

/**
 * Formats an array of sorted 1-indexed steps into idiomatic .loom <where> notation:
 * - Uses range-stride `a-b/k` when preferRange is true (hat/openhat) and steps form an
 *   arithmetic progression of >= 4 hits with stride k in {1, 2, 4, 8, 16}.
 * - Otherwise comma-separated step numbers `1,9,17,25` (so kicks/snares/bass list their steps).
 */
function formatWhereSteps(steps, maxStep, preferRange) {
  if (steps.length === 1) return String(steps[0]);
  if (preferRange && steps.length >= 4) {
    const d = steps[1] - steps[0];
    if (d === 1 || d === 2 || d === 4 || d === 8 || d === 16) {
      let isArith = true;
      for (let i = 2; i < steps.length; i++) {
        if (steps[i] - steps[i - 1] !== d) { isArith = false; break; }
      }
      if (isArith) {
        const first = steps[0];
        const last = steps[steps.length - 1];
        const rangeEnd = (last + d > maxStep) ? maxStep : last;
        return d === 1 ? `${first}-${rangeEnd}` : `${first}-${rangeEnd}/${d}`;
      }
    }
  }
  return steps.join(',');
}

/**
 * Exports what Synth Radio channel `channelIdx` (0..145) at `seed` plays for `barsOpt` bars
 * (default: 4 or 8 bars, chosen deterministically from seed) as a comment-free .loom score.
 */
function exportChannelLoom(channelIdx, seed, barsOpt) {
  const channel = CHANNELS[channelIdx];
  if (!channel) throw new Error(`Invalid channelIdx ${channelIdx}`);
  const song = generateSong(channel, seed);
  const B = barsOpt || (((seed >>> 0) % 3 === 0) ? 4 : 8);
  const windowSteps = B * 16;

  // Pick a bar-aligned window inside the generated song deterministically from seed
  const totalSongSteps = Math.max(windowSteps, song.totalSteps || 64);
  const maxStartBar = Math.max(0, Math.floor((totalSongSteps - windowSteps) / 16));
  const startBar = maxStartBar > 0 ? (((seed >>> 0) * 2654435761 >>> 0) % (maxStartBar + 1)) : 0;
  const winStartStep = startBar * 16;
  const winEndStep = winStartStep + windowSteps;

  const bpm = Math.max(40, Math.min(240, Math.round(song.bpm)));
  const spb = 60 / song.bpm;
  const stepDur = spb / 4;
  const auraScale = SYNTH_RADIO_TO_AURALOOM_SCALE[song.scaleName] || 'minor';
  const useFlats = FLAT_SCALES.has(auraScale) && (FLAT_ROOTS.has(song.key) || ((seed >>> 0) & 1) === 0);
  const rootName = (useFlats ? NOTE_NAMES_FLAT : NOTE_NAMES_SHARP)[song.key % 12];

  const rawLeadVoice = channel.leadVoice || (channel.voices && channel.voices[0]) || 'synth_lead_saw';
  const headerVoice = SYNTH_RADIO_TO_AURALOOM_VOICE[rawLeadVoice] || 'fmLead';
  const spaceVal = Math.max(0, Math.min(100, Math.round((channel.reverb != null ? channel.reverb : 0.3) * 100)));
  const crunchVal = Math.max(0, Math.min(100, Math.round(
    (channel.cluster === 'rock' || /metal|punk|grunge|industrial|acid|phonk|harsh/.test(channel.id || ''))
      ? 32
      : (channel.cluster === 'hiphop' || channel.cluster === 'retro' ? 16 : 6)
  )));

  // Select column-alignment style deterministically from seed to match AuraLoom writeScore & crew scores:
  //   0: Ed/Vael style (padEnd(7) for single steps, padEnd(15) for short lists, padEnd(28) for drum/long lists)
  //   1: Kehai peepers style (padEnd(4) for tone, padEnd(16) for pluck, padEnd(28) for drums)
  //   2: AuraLoom writeScore style (padEnd(15) for all lines)
  //   3: Compact single-space style
  const alignStyle = (seed >>> 0) % 4;
  function padWhere(wStr, kind) {
    if (alignStyle === 3) return wStr + ' ';
    if (alignStyle === 2) return wStr.length < 14 ? wStr.padEnd(14) + ' ' : wStr + ' ';
    if (alignStyle === 1) {
      const width = kind === 'drum' ? 28 : (kind === 'pluck' ? 16 : 4);
      return wStr.length < width ? wStr.padEnd(width) : wStr + ' ';
    }
    // alignStyle === 0 (Ed/Vael style: 7 chars for short <where>, 15 for medium, 28 for long)
    if (wStr.length < 7) return wStr.padEnd(7);
    if (wStr.length < 15) return wStr.padEnd(15);
    if (wStr.length < 28) return wStr.padEnd(28);
    return wStr + ' ';
  }

  const outLines = [
    `bpm ${bpm}`,
    `bars ${B}`,
    `key ${rootName} ${auraScale}`,
    `voice ${headerVoice}`,
    `space ${spaceVal}`,
    `crunch ${crunchVal}`,
    ``
  ];

  // Check if song has texture/atmosphere events
  const hasTexture = song.timeline.some(ev => ev.kind === 'texture');
  if (hasTexture && ((seed >>> 0) % 2 === 0)) {
    const noiseNote = `${rootName}${(seed & 1) ? 7 : 8}`;
    const noiseLoud = Math.max(12, Math.min(36, Math.round(spaceVal * 0.4 + 10)));
    outLines.push(`${padWhere('1', 'noise')}noise ${noiseNote} ${windowSteps} ${noiseLoud} width=2`);
    outLines.push(``);
  }

  function collectWindowEvents(wStart, wEnd) {
    const drums = [];
    const notes = [];
    for (const ev of song.timeline) {
      if (ev.kind === 'drum') {
        const gs = ev.gridStep != null ? ev.gridStep : Math.round(ev.time / stepDur);
        if (gs >= wStart && gs < wEnd) {
          drums.push({ step: gs - wStart + 1, drum: ev.drum, vel: (ev.params && ev.params.vel) || 0.7 });
        }
      } else if (ev.kind === 'note') {
        const gs = ev.gridStep != null ? ev.gridStep : Math.round(ev.time / stepDur);
        if (gs >= wStart && gs < wEnd) {
          notes.push({
            step: gs - wStart + 1,
            voice: ev.voice,
            midi: ev.midi,
            durSteps: Math.max(1, Math.min(windowSteps, Math.round(ev.dur / stepDur))),
            vel: (ev.params && ev.params.vel) || 0.6,
            role: ev.role || 'lead'
          });
        } else if (gs === 0 && ev.dur >= song.duration * 0.8) {
          notes.push({
            step: 1,
            voice: ev.voice,
            midi: ev.midi,
            durSteps: windowSteps,
            vel: (ev.params && ev.params.vel) || 0.3,
            role: 'chord'
          });
        }
      }
    }
    return { drums, notes };
  }

  let { drums, notes } = collectWindowEvents(winStartStep, winEndStep);
  if (drums.length + notes.length === 0) {
    for (let b = 0; b <= maxStartBar; b++) {
      const cand = collectWindowEvents(b * 16, b * 16 + windowSteps);
      if (cand.drums.length + cand.notes.length > 0) {
        drums = cand.drums;
        notes = cand.notes;
        break;
      }
    }
  }
  if (drums.length + notes.length === 0) {
    notes.push({
      step: 1,
      voice: rawLeadVoice,
      midi: 48 + (song.key % 12) + 12,
      durSteps: Math.min(16, windowSteps),
      vel: 0.45,
      role: 'lead'
    });
  }

  // ── 1. DRUMS (kicks/snares as step lists 1,9,17,25; regular hats as 1-64/2) ──
  if (drums.length > 0) {
    const drumStepMap = new Map();
    for (const d of drums) {
      const th = SYNTH_RADIO_TO_AURALOOM_DRUM[d.drum] || 'hat';
      const loud = Math.max(18, Math.min(95, Math.round((d.vel * 100) / 5) * 5));
      const key = `${th}:${d.step}`;
      const prev = drumStepMap.get(key);
      if (prev == null || loud > prev) drumStepMap.set(key, loud);
    }
    const drumOrder = ['kick', 'snare', 'hat', 'openhat'];
    const drumGroups = new Map();
    for (const [k, loud] of drumStepMap.entries()) {
      const [th, stStr] = k.split(':');
      const gk = `${th}:${loud}`;
      if (!drumGroups.has(gk)) drumGroups.set(gk, { th, loud, steps: [] });
      drumGroups.get(gk).steps.push(Number(stStr));
    }
    const sortedGroups = [...drumGroups.values()].sort((a, b) => {
      const da = drumOrder.indexOf(a.th) - drumOrder.indexOf(b.th);
      return da !== 0 ? da : b.loud - a.loud;
    });
    for (const g of sortedGroups) {
      g.steps.sort((a, b) => a - b);
      const preferRange = (g.th === 'hat' || g.th === 'openhat');
      const whereStr = formatWhereSteps(g.steps, windowSteps, preferRange);
      if (whereStr.includes(',') && g.steps.length > 16) {
        for (let i = 0; i < g.steps.length; i += 16) {
          const chunk = g.steps.slice(i, i + 16).join(',');
          outLines.push(`${padWhere(chunk, 'drum')}${g.th} ${g.loud}`);
        }
      } else {
        outLines.push(`${padWhere(whereStr, 'drum')}${g.th} ${g.loud}`);
      }
    }
    outLines.push(``);
  }

  // ── 2. BASS (laid out in bars) & CHORD PADS / COMP / ARPEGGIO (pluck) ──
  const pluckEvents = notes.filter(n => n.role === 'bass' || n.role === 'chord' || n.role === 'arp' || n.role === 'color');
  const leadRawEvents = notes.filter(n => n.role === 'lead');

  if (pluckEvents.length > 0) {
    const simMap = new Map();
    for (const ev of pluckEvents) {
      const auraVoice = SYNTH_RADIO_TO_AURALOOM_VOICE[ev.voice] || (ev.role === 'bass' ? 'fmBass' : 'fmKeys');
      const loud = Math.max(20, Math.min(92, Math.round((ev.vel * 100) / 4) * 4));
      const k = `${ev.role}:${ev.step}:${auraVoice}:${ev.durSteps}`;
      if (!simMap.has(k)) simMap.set(k, { role: ev.role, step: ev.step, voice: auraVoice, durSteps: ev.durSteps, loud, midis: [] });
      const entry = simMap.get(k);
      const mRound = Math.round(ev.midi);
      if (!entry.midis.includes(mRound)) entry.midis.push(mRound);
      if (loud > entry.loud) entry.loud = loud;
    }
    // Group bass within each bar (barIdx = Math.floor((step-1)/16)) so bass is laid out in bars!
    const patternGroupMap = new Map();
    for (const entry of simMap.values()) {
      entry.midis.sort((a, b) => a - b);
      const noteTok = entry.midis.slice(0, 4).map(m => midiToLoomNote(m, useFlats)).join('+');
      const barGroup = entry.role === 'bass' ? Math.floor((entry.step - 1) / 16) : 0;
      const pk = `${entry.role}|${barGroup}|${noteTok}|${entry.durSteps}|${entry.loud}|${entry.voice}`;
      if (!patternGroupMap.has(pk)) {
        patternGroupMap.set(pk, {
          role: entry.role,
          barGroup,
          noteTok,
          durSteps: entry.durSteps,
          loud: entry.loud,
          voice: entry.voice,
          steps: []
        });
      }
      patternGroupMap.get(pk).steps.push(entry.step);
    }
    const roleOrder = { bass: 0, chord: 1, arp: 2, color: 3 };
    const groupedPlucks = [...patternGroupMap.values()].sort((a, b) => {
      const dr = (roleOrder[a.role] || 0) - (roleOrder[b.role] || 0);
      if (dr !== 0) return dr;
      if (a.barGroup !== b.barGroup) return a.barGroup - b.barGroup;
      return a.steps[0] - b.steps[0];
    });
    for (const gp of groupedPlucks) {
      gp.steps.sort((a, b) => a - b);
      const uniqueSteps = [...new Set(gp.steps)];
      for (let i = 0; i < uniqueSteps.length; i += 12) {
        const whereStr = uniqueSteps.slice(i, i + 12).join(',');
        outLines.push(`${padWhere(whereStr, 'pluck')}pluck ${gp.noteTok} ${gp.durSteps} ${gp.loud} voice=${gp.voice}`);
      }
    }
    outLines.push(``);
  }

  // ── 3. LEAD MELODY (tone, with approach glides N1>N2) ──
  if (leadRawEvents.length > 0) {
    leadRawEvents.sort((a, b) => a.step - b.step || a.midi - b.midi);
    const leadLines = [];
    let i = 0;
    while (i < leadRawEvents.length) {
      const cur = leadRawEvents[i];
      const auraVoice = SYNTH_RADIO_TO_AURALOOM_VOICE[cur.voice] || headerVoice;
      const loud = Math.max(30, Math.min(92, Math.round((cur.vel * 100) / 2) * 2));
      if (
        cur.durSteps <= 1 &&
        i + 1 < leadRawEvents.length &&
        leadRawEvents[i + 1].step === cur.step + 1 &&
        (SYNTH_RADIO_TO_AURALOOM_VOICE[leadRawEvents[i + 1].voice] || headerVoice) === auraVoice &&
        Math.round(leadRawEvents[i + 1].midi) !== Math.round(cur.midi)
      ) {
        const nxt = leadRawEvents[i + 1];
        const n1 = midiToLoomNote(cur.midi, useFlats);
        const n2 = midiToLoomNote(nxt.midi, useFlats);
        const glideLen = Math.max(2, Math.min(windowSteps, nxt.durSteps + 1));
        const glideLoud = Math.max(loud, Math.min(92, Math.round((nxt.vel * 100) / 2) * 2));
        const voiceSuffix = auraVoice !== headerVoice ? ` voice=${auraVoice}` : '';
        leadLines.push(`${padWhere(String(cur.step), 'tone')}tone ${n1}>${n2} ${glideLen} ${glideLoud}${voiceSuffix}`);
        i += 2;
        continue;
      }
      const nTok = midiToLoomNote(cur.midi, useFlats);
      const voiceSuffix = auraVoice !== headerVoice ? ` voice=${auraVoice}` : '';
      leadLines.push(`${padWhere(String(cur.step), 'tone')}tone ${nTok} ${cur.durSteps} ${loud}${voiceSuffix}`);
      i++;
    }
    for (const l of leadLines) outLines.push(l);
  }

  while (outLines.length && outLines[outLines.length - 1] === '') outLines.pop();
  return outLines.join('\n') + '\n';
}

/* ════════════════════════════════════════════════════════════════
   5. BYTE-ROLE CLASSIFIER FOR .LOOM SCORES
   ════════════════════════════════════════════════════════════════ */

const HEADER_KEYWORDS = new Set(['bpm', 'bars', 'key', 'voice', 'ring', 'space', 'crunch', 'mode']);
const PITCH_TOKEN_RE = /^[A-Ga-g][#♯b♭]?-?\dq?(?:[+>][A-Ga-g][#♯b♭]?-?\dq?)*$/;

/**
 * Classifies every byte of a .loom buffer (`Uint8Array` or `Buffer` or `string`) into:
 *   0 = other (headers, thread/instrument names, lengths, velocities, voice=..., whitespace, newlines)
 *   1 = step  (step-column bytes: digits, `,`, `-`, `/`, `.`, `s` in <where> before the thread word)
 *   2 = pitch (pitch tokens `D2`, `A#3`, `Bb4`, chords `F4+A4+C5`, glides `D5>E5` on event lines)
 *
 * Returns a Uint8Array of length `u8.length`.
 */
function classifyLoomByteRoles(input) {
  const u8 = typeof input === 'string' ? Buffer.from(input, 'utf8') : input;
  const roles = new Uint8Array(u8.length);
  const N = u8.length;
  let lineStart = 0;

  while (lineStart < N) {
    let lineEnd = lineStart;
    while (lineEnd < N && u8[lineEnd] !== 10) lineEnd++; // '\n'

    let codeEnd = lineEnd;
    if (codeEnd > lineStart && u8[codeEnd - 1] === 13) codeEnd--; // '\r'
    for (let i = lineStart; i < codeEnd; i++) {
      if (u8[i] === 35 && (i === lineStart || u8[i - 1] === 32 || u8[i - 1] === 9)) {
        codeEnd = i;
        break;
      }
    }

    const spans = [];
    let p = lineStart;
    while (p < codeEnd) {
      while (p < codeEnd && (u8[p] === 32 || u8[p] === 9)) p++;
      if (p >= codeEnd) break;
      const a = p;
      while (p < codeEnd && u8[p] !== 32 && u8[p] !== 9) p++;
      spans.push([a, p]);
    }

    if (spans.length >= 2) {
      const firstWord = Buffer.from(u8.subarray(spans[0][0], spans[0][1])).toString('utf8').toLowerCase();
      if (!HEADER_KEYWORDS.has(firstWord)) {
        const secondWord = Buffer.from(u8.subarray(spans[1][0], spans[1][1])).toString('utf8').toLowerCase();
        if (THREADS.includes(secondWord)) {
          for (let j = spans[0][0]; j < spans[0][1]; j++) {
            roles[j] = 1; // step role
          }
          for (let ti = 2; ti < spans.length; ti++) {
            const tokStr = Buffer.from(u8.subarray(spans[ti][0], spans[ti][1])).toString('utf8');
            if (PITCH_TOKEN_RE.test(tokStr)) {
              for (let j = spans[ti][0]; j < spans[ti][1]; j++) {
                roles[j] = 2; // pitch role
              }
            }
          }
        }
      }
    }

    lineStart = lineEnd + 1;
  }

  return roles;
}

/* ════════════════════════════════════════════════════════════════
   6. CHANNEL SPLIT, 10,000-EXPORT VALIDATION, CORPUS GENERATION & LEAK CHECK
   ════════════════════════════════════════════════════════════════ */

const SPLIT_SEED = 20261111;

function getChannelSplits(seed = SPLIT_SEED) {
  const rng = mulberry32(seed);
  const indices = Array.from({ length: CHANNELS.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
  }
  const trainChannels = indices.slice(0, 110);
  const valChannels   = indices.slice(110, 120);
  const testChannels  = indices.slice(120, 146); // 26 held-out channels
  return { seed, trainChannels, valChannels, testChannels };
}

function stripLoomComments(text) {
  const out = [];
  for (const raw of text.split(/\r?\n/)) {
    const stripped = raw.replace(/(^|\s)#.*/, '').trimRight();
    if (!stripped.trim()) {
      if (out.length > 0 && out[out.length - 1] !== '') out.push('');
    } else {
      out.push(stripped);
    }
  }
  while (out.length && out[0] === '') out.shift();
  while (out.length && out[out.length - 1] === '') out.pop();
  return out.join('\n') + '\n';
}

function computeLeakStats(trainStr, train16Grams, scoreStr) {
  const N = 16;
  const totalGrams = Math.max(0, scoreStr.length - N + 1);
  let matchedOccurrences = 0;
  const scoreUnique = new Set();
  const matchedUnique = new Set();
  for (let i = 0; i <= scoreStr.length - N; i++) {
    const g = scoreStr.slice(i, i + N);
    scoreUnique.add(g);
    if (train16Grams.has(g)) {
      matchedOccurrences++;
      matchedUnique.add(g);
    }
  }
  let bestLen = 0;
  let bestSub = '';
  for (let i = 0; i < scoreStr.length; i++) {
    if (scoreStr.length - i <= bestLen) break;
    const probe = scoreStr.slice(i, i + bestLen + 1);
    if (trainStr.indexOf(probe) !== -1) {
      let L = bestLen + 1;
      while (i + L < scoreStr.length && trainStr.indexOf(scoreStr.slice(i, i + L + 1)) !== -1) {
        L++;
      }
      bestLen = L;
      bestSub = scoreStr.slice(i, i + L);
    }
  }
  return {
    scoreBytes: Buffer.byteLength(scoreStr, 'utf8'),
    lcsLength: bestLen,
    lcsString: JSON.stringify(bestSub),
    total16Grams: totalGrams,
    matched16GramOccurrences: matchedOccurrences,
    occurrenceOverlapFrac: totalGrams > 0 ? matchedOccurrences / totalGrams : 0,
    unique16Grams: scoreUnique.size,
    matchedUnique16Grams: matchedUnique.size,
    uniqueOverlapFrac: scoreUnique.size > 0 ? matchedUnique.size / scoreUnique.size : 0
  };
}

function validate10000Exports() {
  const total = 10000;
  let passed = 0;
  let totalWoven = 0;
  const failures = [];
  const t0 = Date.now();
  for (let i = 0; i < total; i++) {
    const chIdx = i % CHANNELS.length;
    const seed = 100000 + i * 97;
    const score = exportChannelLoom(chIdx, seed);
    const res = weaveScore(score, (i % 20 === 0));
    if (res.errs.length === 0 && res.woven > 0) {
      passed++;
      totalWoven += res.woven;
    } else {
      failures.push({ i, chIdx, channelId: CHANNELS[chIdx].id, seed, errs: res.errs });
      if (failures.length >= 10) break;
    }
  }
  const elapsedSec = ((Date.now() - t0) / 1000).toFixed(2);
  return { total, passed, failed: total - passed, totalWoven, elapsedSec, failures };
}

function generateAllLoomCorpora(outDir = path.resolve(__dirname, '..')) {
  console.log('1. Validating 10,000 Synth Radio -> .loom exports with AuraLoom weaveScore...');
  const valRes = validate10000Exports();
  console.log(`   -> ${valRes.passed} / ${valRes.total} valid (${valRes.totalWoven} woven events, ${valRes.elapsedSec}s)`);
  if (valRes.failed > 0) {
    console.error('Failures:', JSON.stringify(valRes.failures, null, 2));
    process.exit(1);
  }

  const splits = getChannelSplits(SPLIT_SEED);
  console.log(`2. Channel splits (seed=${splits.seed}): train=${splits.trainChannels.length}, val=${splits.valChannels.length}, unseen_test=${splits.testChannels.length}`);

  const targetTrainBytes = 2000000;
  const trainChunks = [];
  let trainBytes = 0;
  let trainCount = 0;
  while (trainBytes < targetTrainBytes) {
    const chIdx = splits.trainChannels[trainCount % splits.trainChannels.length];
    const seed = 10000 + trainCount * 31;
    const score = exportChannelLoom(chIdx, seed);
    trainChunks.push(score);
    trainBytes += Buffer.byteLength(score, 'utf8') + 1;
    trainCount++;
  }
  const trainText = trainChunks.join('\n');
  fs.writeFileSync(path.join(outDir, 'loom-train.txt'), trainText, 'utf8');

  const targetValBytes = 200000;
  const valChunks = [];
  let valBytes = 0;
  let valCount = 0;
  while (valBytes < targetValBytes) {
    const chIdx = splits.valChannels[valCount % splits.valChannels.length];
    const seed = 500000 + valCount * 37;
    const score = exportChannelLoom(chIdx, seed);
    valChunks.push(score);
    valBytes += Buffer.byteLength(score, 'utf8') + 1;
    valCount++;
  }
  const valText = valChunks.join('\n');
  fs.writeFileSync(path.join(outDir, 'loom-val.txt'), valText, 'utf8');

  const unseenChannelsData = [];
  const unseenAllChunks = [];
  for (let ci = 0; ci < splits.testChannels.length; ci++) {
    const chIdx = splits.testChannels[ci];
    const chObj = CHANNELS[chIdx];
    const chScores = [];
    for (let k = 0; k < 2; k++) {
      const seed = 900000 + ci * 100 + k * 13;
      const score = exportChannelLoom(chIdx, seed, k === 0 ? 4 : 8);
      const v = weaveScore(score, true);
      if (v.errs.length > 0) throw new Error(`Unseen test score invalid: ${v.errs.join('; ')}`);
      chScores.push(score);
      unseenAllChunks.push(score);
    }
    const chText = chScores.join('\n');
    unseenChannelsData.push({
      channelIdx: chIdx,
      id: chObj.id,
      name: chObj.name,
      cluster: chObj.cluster,
      bytes: Buffer.byteLength(chText, 'utf8'),
      text: chText
    });
  }
  const unseenText = unseenAllChunks.join('\n');
  fs.writeFileSync(path.join(outDir, 'loom-test-unseen.txt'), unseenText, 'utf8');
  fs.writeFileSync(path.join(outDir, 'loom-test-unseen.json'), JSON.stringify(unseenChannelsData, null, 2), 'utf8');

  const crewOrder = [
    { id: 'peepers', file: 'peepers.loom.txt', author: 'Kehai' },
    { id: 'hollow', file: 'hollow.loom.txt', author: 'Ed' },
    { id: 'hollow-2', file: 'hollow-2.loom.txt', author: 'Ed' },
    { id: 'threshold', file: 'threshold.loom.txt', author: 'Vael' },
    { id: 'understory', file: 'understory.loom.txt', author: 'Ed' },
    { id: 'understory-morning', file: 'understory-morning.loom.txt', author: 'Ed' },
    { id: 'understory-rest', file: 'understory-rest.loom.txt', author: 'Ed' },
    { id: 'understory-trouble', file: 'understory-trouble.loom.txt', author: 'Ed' }
  ];
  const crewDir = path.join(outDir, 'rhythm_pack/rhythm-pack/auraloom/scores');
  const crewData = [];
  const crewChunks = [];

  console.log('3. Building 16-gram index on loom-train.txt and running crew leak checks...');
  const train16Grams = new Set();
  for (let i = 0; i <= trainText.length - 16; i++) {
    train16Grams.add(trainText.slice(i, i + 16));
  }

  for (const item of crewOrder) {
    const raw = fs.readFileSync(path.join(crewDir, item.file), 'utf8');
    const stripped = stripLoomComments(raw);
    const v = weaveScore(stripped, true);
    if (v.errs.length > 0) {
      throw new Error(`Crew score ${item.id} failed weaveScore after comment stripping: ${v.errs.join('; ')}`);
    }
    const leak = computeLeakStats(trainText, train16Grams, stripped);
    const roles = classifyLoomByteRoles(stripped);
    let stepBytes = 0, pitchBytes = 0, otherBytes = 0;
    for (let i = 0; i < roles.length; i++) {
      if (roles[i] === 1) stepBytes++;
      else if (roles[i] === 2) pitchBytes++;
      else otherBytes++;
    }
    crewData.push({
      id: item.id,
      author: item.author,
      file: item.file,
      rawBytes: Buffer.byteLength(raw, 'utf8'),
      bytes: Buffer.byteLength(stripped, 'utf8'),
      woven: v.woven,
      bars: v.bars,
      bpm: v.bpm,
      roleBytes: { step: stepBytes, pitch: pitchBytes, other: otherBytes },
      leak,
      text: stripped
    });
    crewChunks.push(stripped);
  }
  const crewText = crewChunks.join('\n');
  fs.writeFileSync(path.join(outDir, 'loom-test-crew.txt'), crewText, 'utf8');
  fs.writeFileSync(path.join(outDir, 'loom-test-crew.json'), JSON.stringify(crewData, null, 2), 'utf8');

  function countRoles(txt) {
    const r = classifyLoomByteRoles(txt);
    let step = 0, pitch = 0, other = 0;
    for (let i = 0; i < r.length; i++) {
      if (r[i] === 1) step++;
      else if (r[i] === 2) pitch++;
      else other++;
    }
    return { total: r.length, step, pitch, other };
  }

  const meta = {
    splitSeed: SPLIT_SEED,
    totalChannels: CHANNELS.length,
    trainChannelCount: splits.trainChannels.length,
    valChannelCount: splits.valChannels.length,
    unseenTestChannelCount: splits.testChannels.length,
    trainChannels: splits.trainChannels.map(i => ({ index: i, id: CHANNELS[i].id, name: CHANNELS[i].name, cluster: CHANNELS[i].cluster })),
    valChannels: splits.valChannels.map(i => ({ index: i, id: CHANNELS[i].id, name: CHANNELS[i].name, cluster: CHANNELS[i].cluster })),
    unseenTestChannels: splits.testChannels.map(i => ({ index: i, id: CHANNELS[i].id, name: CHANNELS[i].name, cluster: CHANNELS[i].cluster })),
    voiceMapping: SYNTH_RADIO_TO_AURALOOM_VOICE,
    scaleMapping: SYNTH_RADIO_TO_AURALOOM_SCALE,
    drumMapping: SYNTH_RADIO_TO_AURALOOM_DRUM,
    validation10k: valRes,
    corpusStats: {
      trainScores: trainCount,
      trainRoleBytes: countRoles(trainText),
      valScores: valCount,
      valRoleBytes: countRoles(valText),
      unseenScores: unseenAllChunks.length,
      unseenRoleBytes: countRoles(unseenText),
      crewScores: crewData.length,
      crewRoleBytes: countRoles(crewText)
    },
    crewScores: crewData.map(c => ({
      id: c.id,
      author: c.author,
      rawBytes: c.rawBytes,
      bytes: c.bytes,
      woven: c.woven,
      bars: c.bars,
      bpm: c.bpm,
      roleBytes: c.roleBytes,
      leak: c.leak
    }))
  };

  fs.writeFileSync(path.join(outDir, 'loom-corpus-meta.json'), JSON.stringify(meta, null, 2), 'utf8');
  console.log('4. Corpus summary:');
  console.log(`   - loom-train.txt:       ${meta.corpusStats.trainRoleBytes.total} bytes (${trainCount} scores)`);
  console.log(`   - loom-val.txt:         ${meta.corpusStats.valRoleBytes.total} bytes (${valCount} scores)`);
  console.log(`   - loom-test-unseen.txt: ${meta.corpusStats.unseenRoleBytes.total} bytes (${unseenAllChunks.length} scores across 26 channels)`);
  console.log(`   - loom-test-crew.txt:   ${meta.corpusStats.crewRoleBytes.total} bytes (8 crew scores, comments stripped)`);
  console.log('5. Crew leak check:');
  for (const c of meta.crewScores) {
    console.log(`   - ${c.id.padEnd(20)} (${String(c.bytes).padStart(4)} B): LCS=${String(c.leak.lcsLength).padStart(2)} B (${c.leak.lcsString}), 16-gram unique overlap=${(c.leak.uniqueOverlapFrac * 100).toFixed(2)}% (${c.leak.matchedUnique16Grams}/${c.leak.unique16Grams})`);
  }
  return meta;
}

if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.includes('--validate-only')) {
    const res = validate10000Exports();
    console.log(JSON.stringify(res, null, 2));
    process.exit(res.failed === 0 ? 0 : 1);
  } else {
    generateAllLoomCorpora();
  }
}

module.exports = {
  CHANNELS,
  CLUSTERS,
  SYNTH_RADIO_TO_AURALOOM_VOICE,
  SYNTH_RADIO_TO_AURALOOM_SCALE,
  SYNTH_RADIO_TO_AURALOOM_DRUM,
  generateSong,
  exportChannelLoom,
  weaveScore,
  parseNote,
  parseWhere,
  classifyLoomByteRoles,
  getChannelSplits,
  stripLoomComments,
  validate10000Exports,
  generateAllLoomCorpora
};