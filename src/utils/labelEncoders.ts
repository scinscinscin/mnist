import { Matrix } from "../matrix";

export const onehotEncoder = (n_classes: number) => {
  return (x: number) => {
    const array = Array(n_classes).fill(0);
    array[x] = 1;
    return Matrix.fromData([array]);
  };
};

export const directEncoder = () => {
  return (x: number) => {
    return Matrix.fromData([[x]]);
  };
};
