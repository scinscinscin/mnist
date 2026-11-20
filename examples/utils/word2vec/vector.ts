/**
 * A word and its vector representation.
 */
export class WordVector {
  constructor(
    /** the word this vector represents (`null` for computed vectors such as sums) */
    public readonly word: string | null,
    /** the elements of the word vector */
    public readonly values: Float32Array,
  ) {}

  /**
   * Add another word vector to this word vector.
   * @returns a new WordVector holding the result of the addition
   */
  add(other: WordVector): WordVector {
    const values = new Float32Array(this.values.length);
    for (let i = 0; i < values.length; i++) {
      values[i] = this.values[i] + other.values[i];
    }
    return new WordVector(null, values);
  }

  /**
   * Subtract another word vector from this word vector.
   * @returns a new WordVector holding the result of the subtraction
   */
  subtract(other: WordVector): WordVector {
    const values = new Float32Array(this.values.length);
    for (let i = 0; i < values.length; i++) {
      values[i] = this.values[i] - other.values[i];
    }
    return new WordVector(null, values);
  }

  to_number_array(): number[] {
    return [...this.values];
  }
}
