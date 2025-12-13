export class RNG {
  m = 0x80000000; // 2**31;
  a = 1103515245;
  c = 12345;
  seed = 0;

  constructor(seed: number) {
    this.seed = seed ? seed : Math.floor(Math.random() * (this.m - 1));
  }

  nextInt() {
    this.seed = (this.a * this.seed + this.c) % this.m;
    return this.seed;
  }

  nextFloat() {
    // returns in range [0,1]
    return this.nextInt() / (this.m - 1);
  }

  nextRange(start: number, end: number) {
    // returns in range [start, end): including start, excluding end
    // can't modulu nextInt because of weak randomness in lower bits
    var rangeSize = end - start;
    var randomUnder1 = this.nextInt() / this.m;
    return start + Math.floor(randomUnder1 * rangeSize);
  }

  choice(array: string[]) {
    return array[this.nextRange(0, array.length)];
  }
}
