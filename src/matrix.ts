export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;

  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Matrix {
  arr = [] as number[][];

  private constructor(rows: number, cols: number, get_number: () => number) {
    let arr = [] as number[][];

    for (let i = 0; i < rows; i++) {
      arr[i] = [];
      for (let j = 0; j < cols; j++) arr[i][j] = get_number();
    }

    this.arr = arr;
  }

  static init(rows: number, cols: number) {
    return new Matrix(rows, cols, () => 0);
  }

  static from_existing(arr: number[][]) {
    const m = new Matrix(arr.length, arr[0].length ?? 0, () => 0);
    m.arr = arr;
    return m;
  }

  static from_random(rows: number, cols: number, seed: number) {
    const rng = mulberry32(seed);
    return new Matrix(rows, cols, () => rng());
  }

  // uniform in [-scale, scale]; scale is typically 1 / sqrt(input) for weight init
  static from_uniform(rows: number, cols: number, seed: number, scale: number) {
    const rng = mulberry32(seed);
    return new Matrix(rows, cols, () => (rng() * 2 - 1) * scale);
  }

  get cols() {
    return this.arr.length ? this.arr[0].length : 0;
  }

  get rows() {
    return this.arr.length;
  }

  add(other: Matrix) {
    if (this.cols !== other.cols || this.rows !== other.rows) {
      throw new Error(
        `Cannot add matrices of different dimensions: ${this.rows}x${this.cols} + ${other.rows}x${other.cols}`,
      );
    }

    const result = Matrix.init(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++) result.arr[i][j] = this.arr[i][j] + other.arr[i][j];
    return result;
  }

  multiply(b: Matrix) {
    if (this.cols !== b.rows) {
      throw new Error(
        `Cannot multiply matrices with incompatible dimensions: ${this.rows}x${this.cols} * ${b.rows}x${b.cols}`,
      );
    }

    const result = Matrix.init(this.rows, b.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < b.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) sum += this.arr[i][k] * b.arr[k][j];
        result.arr[i][j] = sum;
      }
    }

    return result;
  }

  multiply_scalar(scalar: number) {
    const result = Matrix.init(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.cols; j++) result.arr[i][j] = this.arr[i][j] * scalar;
    return result;
  }

  append_column(x: number) {
    const result = Matrix.init(this.rows, this.cols + 1);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) result.arr[i][j] = this.arr[i][j];
      result.arr[i][this.cols] = x;
    }
    return result;
  }

  transpose() {
    const result = Matrix.init(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.cols; j++) result.arr[j][i] = this.arr[i][j];
    return result;
  }

  map(fn: (x: number) => number) {
    const result = Matrix.init(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.cols; j++) result.arr[i][j] = fn(this.arr[i][j]);
    return result;
  }

  print() {
    return this.arr.map((row) => row.map((x) => x.toFixed(3)).join(" ")).join("\n");
  }

  to_json() {
    return { rows: this.rows, cols: this.cols, data: this.arr };
  }
}
