import path from "node:path";
import { WordVector } from "./vector";
import fs from "node:fs";

/** A word and its distance (cosine similarity) to a query vector. */
export interface WordDist {
  word: string;
  dist: number;
}

/** A loaded word2vec model. */
export interface WordModel {
  /** number of words in the vocabulary */
  words: number;
  /** dimensionality of the word vectors */
  size: number;

  /** Get the vector for a single word, or `null` if the word is out of dictionary. */
  get_vector(word: string): WordVector | null;

  /** Get all word vectors, or only those whose word is contained in `words`. */
  get_vectors(words?: string[]): WordVector[];

  /** Dot-product similarity of two words, or `null` if at least one is out of dictionary. */
  similarity(word1: string, word2: string): number | null;

  /** The word most similar to the given vector. */
  get_nearest_word(vec: WordVector | Float32Array): WordDist;

  /** The `n` words most similar to the given vector (default 10). */
  get_nearest_words(vec: WordVector | Float32Array, n?: number): WordDist[];

  /**
   * The `n` words most similar to a space-separated phrase (default 40).
   * Returns `null` when none of the phrase words are in the dictionary.
   */
  most_similar(phrase: string, n?: number): WordDist[] | null;

  /**
   * Words most similar to the vector `pair[1] - pair[0] + word` (default 40).
   * The words of `pair` and `word` must all be in the dictionary.
   */
  analogy(word: string, pair: [string, string], n?: number): WordDist[];
}

/**
 * Load a word2vec model from a file in text format or binary `.bin` format.
 */
export function load_model(file: string): WordModel {
  const resolved = path.resolve(process.cwd(), file);
  const raw = fs.readFileSync(resolved);
  const { words, size } = read_header(raw);
  const binary = path.extname(resolved).toLowerCase() === ".bin";
  const vocab = binary ? read_binary(raw, size) : read_txt(raw, size);

  return {
    words,
    size,

    get_vector(word: string): WordVector | null {
      return vocab.find((v) => v.word === word) ?? null;
    },

    get_vectors(filter?: string[]): WordVector[] {
      if (!filter) return vocab;
      return vocab.filter((v) => filter.includes(word_of(v)));
    },

    similarity(word1: string, word2: string): number | null {
      const vecs: Float32Array[] = [];
      for (const v of vocab) if (v.word === word1 || v.word === word2) vecs.push(v.values);

      if (vecs.length === 2) return dot(vecs[0], vecs[1]);
      // Case: word1 and word2 are identical
      else if (vecs.length === 1 && word1 === word2) return 1.0;
      // Case: at least one of the words is not available in the vocabulary
      return null;
    },

    get_nearest_word(vec: WordVector | Float32Array): WordDist {
      const values = normalize(to_values(vec));
      let best_word = "";
      let best_dist = -Infinity;
      for (const v of vocab) {
        const dist = dot(values, v.values);
        if (dist > best_dist) {
          best_dist = dist;
          best_word = word_of(v);
        }
      }
      return { word: best_word, dist: best_dist };
    },

    get_nearest_words(vec: WordVector | Float32Array, n = 10): WordDist[] {
      const values = normalize(to_values(vec));
      const best_words = new Array<string>(n).fill("");
      const best_dists = new Array<number>(n).fill(-1);
      for (const v of vocab) {
        insert_top_n(best_words, best_dists, word_of(v), dot(values, v.values));
      }
      return top_n_to_results(best_words, best_dists);
    },

    most_similar(phrase: string, n = 40): WordDist[] | null {
      const phrase_words = phrase.split(" ").map((word) => ({ word, pos: -1 }));

      // Check whether at least one phrase word is in the dictionary...
      let found = false;
      for (const pw of phrase_words) {
        pw.pos = vocab.findIndex((v) => v.word === pw.word);
        if (pw.pos !== -1) {
          found = true;
        } else {
          console.log(`Out of dictionary word: ${pw.word}`);
        }
      }

      // All words are out-of-dictionary
      if (!found) return null;

      const vec = new Float32Array(size);
      for (const pw of phrase_words) {
        if (pw.pos !== -1) {
          const values = vocab[pw.pos].values;
          for (let a = 0; a < size; a++) {
            vec[a] += values[a];
          }
        }
      }

      // Normalize the summed vector...
      const normalized = normalize(vec);

      const best_words = new Array<string>(n).fill("");
      const best_dists = new Array<number>(n).fill(-1);
      for (let c = 0; c < vocab.length; c++) {
        // Skip words that are part of the phrase
        if (phrase_words.some((pw) => pw.pos === c)) continue;
        insert_top_n(best_words, best_dists, word_of(vocab[c]), dot(normalized, vocab[c].values));
      }
      return top_n_to_results(best_words, best_dists);
    },

    analogy(word: string, pair: [string, string], n = 40): WordDist[] {
      if (typeof word !== "string") throw new TypeError("Word of interest has to be supplied as string.");
      if (!Array.isArray(pair)) throw new TypeError("Word pair has to be supplied in string Array.");

      const phrase = [pair[0], pair[1], word].map((w) => ({ word: w, pos: -1 }));
      for (const pw of phrase) {
        pw.pos = vocab.findIndex((v) => v.word === pw.word);
        if (pw.pos === -1) console.log(`Out of dictionary word: ${pw.word}`);
      }

      if (phrase.some((pw) => pw.pos === -1))
        throw new Error("Words of the analogy must all be in the model vocabulary.");

      // vec = pair[1] - pair[0] + word
      const vec = new Float32Array(size);
      for (let a = 0; a < size; a++)
        vec[a] = vocab[phrase[1].pos].values[a] - vocab[phrase[0].pos].values[a] + vocab[phrase[2].pos].values[a];

      // Normalize the vector...
      const normalized = normalize(vec);

      const best_words = new Array<string>(n).fill("");
      const best_dists = new Array<number>(n).fill(0);
      for (let c = 0; c < vocab.length; c++) {
        // Skip words that are part of the phrase
        if (phrase.some((pw) => pw.pos === c)) continue;
        insert_top_n(best_words, best_dists, word_of(vocab[c]), dot(normalized, vocab[c].values));
      }

      return top_n_to_results(best_words, best_dists);
    },
  };
}

/** Parse the header line `<number of words> <length of vector>`. */
function read_header(raw: Buffer): { words: number; size: number } {
  const newline = raw.indexOf(0x0a);
  const header = raw.subarray(0, newline === -1 ? raw.length : newline).toString("latin1");
  const parts = header.split(/\s+/);
  const words = parseInt(parts[0], 10);
  const size = parseInt(parts[1], 10);
  if (Number.isNaN(words) || Number.isNaN(size))
    throw new Error("First line of file has to be <number of words> <length of vector>.");
  return { words, size };
}

/** Parse a text-format word2vec model. */
function read_txt(raw: Buffer, size: number): WordVector[] {
  const lines = raw.toString("utf8").split(/\r?\n/);
  lines.shift(); // header
  const vocab: WordVector[] = [];
  for (const line of lines) {
    if (!line.trim()) {
      continue;
    }
    const parts = line.split(" ");
    const word = parts[0];
    const values = new Float32Array(size);
    for (let i = 1; i < parts.length; i++) {
      if (parts[i] !== "") {
        values[i - 1] = parseFloat(parts[i]);
      }
    }
    vocab.push(new WordVector(word, normalize(values)));
  }
  return vocab;
}

/** Parse a binary-format word2vec model. */
function read_binary(raw: Buffer, size: number): WordVector[] {
  const vocab: WordVector[] = [];
  let pos = raw.indexOf(0x0a) + 1; // skip header line
  while (pos < raw.length) {
    const space = raw.indexOf(0x20, pos);
    if (space === -1) {
      break;
    }
    const word = raw.subarray(pos, space).toString("utf8").trim();
    const values = new Float32Array(size);
    let off = space + 1;
    for (let i = 0; i < size; i++) {
      values[i] = raw.readFloatLE(off);
      off += 4;
    }
    pos = off;
    vocab.push(new WordVector(word, normalize(values)));
  }
  return vocab;
}

function to_values(vec: WordVector | Float32Array): Float32Array {
  return vec instanceof WordVector ? vec.values : vec;
}

/** The word of a vocabulary vector (vocabulary vectors always carry a word). */
function word_of(v: WordVector): string {
  return v.word ?? "";
}

/** Return a unit-length copy of the given vector. */
function normalize(values: Float32Array): Float32Array {
  let len = 0;
  for (let a = 0; a < values.length; a++) {
    len += values[a] * values[a];
  }
  len = Math.sqrt(len);
  const out = new Float32Array(values.length);
  for (let a = 0; a < values.length; a++) {
    out[a] = values[a] / len;
  }
  return out;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** Insert a candidate into the top-n lists, keeping them sorted by descending distance. */
function insert_top_n(best_words: string[], best_dists: number[], word: string, dist: number): void {
  for (let a = 0; a < best_words.length; a++) {
    if (dist > best_dists[a]) {
      for (let d = best_words.length - 1; d > a; d--) {
        best_dists[d] = best_dists[d - 1];
        best_words[d] = best_words[d - 1];
      }
      best_dists[a] = dist;
      best_words[a] = word;
      return;
    }
  }
}

function top_n_to_results(best_words: string[], best_dists: number[]): WordDist[] {
  return best_words.map((word, i) => ({ word, dist: best_dists[i] }));
}
