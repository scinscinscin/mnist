export class Matrix {
  data: number[][];
  rows: number;
  columns: number;

  private constructor(rows: number, columns: number) {
    this.rows = rows;
    this.columns = columns;
    this.data = Array.from({ length: rows }, () => Array(columns).fill(0));
  }

  setData(data: number[][]) {
    if (data.length !== this.rows || data[0].length !== this.columns) throw new Error("Invalid data");

    this.data = data;
    return this;
  }

  static fromData(data: number[][]) {
    const rows = data.length;
    const columns = data[0].length;
    return new Matrix(rows, columns).setData(data);
  }

  static fromZeros(rows: number, columns: number) {
    return new Matrix(rows, columns);
  }

  static fromRandom(rows: number, columns: number, rngFunction?: () => number) {
    const data = Array.from({ length: rows }, () => Array(columns).fill(0));

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        data[i][j] = (rngFunction ?? Math.random)();
      }
    }

    return Matrix.fromData(data);
  }

  multiplyWith(matrix: Matrix) {
    if (this.columns !== matrix.rows) throw new Error("Invalid matrix");

    const result = Array.from({ length: this.rows }, () => Array(matrix.columns).fill(0));

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        for (let k = 0; k < this.columns; k++) {
          result[i][j] += this.data[i][k] * matrix.data[k][j];
        }
      }
    }

    return Matrix.fromData(result);
  }

  elementWiseMultiplyWith(matrix: Matrix) {
    // if (this.rows !== matrix.rows || this.columns !== matrix.columns) throw new Error("Invalid matrix");

    const result = Array.from({ length: this.rows }, () => Array(this.columns).fill(0));

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        result[i][j] = (this.data[i][j] ?? 1) * (matrix.data[i][j] ?? 1);
      }
    }

    return Matrix.fromData(result);
  }

  addWith(matrix: Matrix) {
    if (this.rows !== matrix.rows || this.columns !== matrix.columns) throw new Error("Invalid matrix");

    const result = Array.from({ length: this.rows }, () => Array(this.columns).fill(0));

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        result[i][j] = this.data[i][j] + matrix.data[i][j];
      }
    }

    return Matrix.fromData(result);
  }

  subtractWith(matrix: Matrix) {
    if (this.rows !== matrix.rows || this.columns !== matrix.columns) throw new Error("Invalid matrix");

    const result = Array.from({ length: this.rows }, () => Array(this.columns).fill(0));

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        result[i][j] = this.data[i][j] - matrix.data[i][j];
      }
    }

    return Matrix.fromData(result);
  }

  mapWith(fn: (value: number) => number) {
    const result = Array.from({ length: this.rows }, () => Array(this.columns).fill(0));

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        result[i][j] = fn(this.data[i][j]);
      }
    }

    return Matrix.fromData(result);
  }

  average() {
    let sum = 0;

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        sum += this.data[i][j];
      }
    }

    return sum / (this.rows * this.columns);
  }

  transpose() {
    const result = Array.from({ length: this.columns }, () => Array(this.rows).fill(0));
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.columns; j++) result[j][i] = this.data[i][j];
    return Matrix.fromData(result);
  }

  dropLastRow() {
    const result = Array.from({ length: this.rows - 1 }, () => Array(this.columns).fill(0));
    for (let i = 0; i < this.rows - 1; i++) for (let j = 0; j < this.columns; j++) result[i][j] = this.data[i][j];
    return Matrix.fromData(result);
  }

  dropLastColumn() {
    const result = Array.from({ length: this.rows }, () => Array(this.columns - 1).fill(0));
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.columns - 1; j++) result[i][j] = this.data[i][j];
    return Matrix.fromData(result);
  }
}
