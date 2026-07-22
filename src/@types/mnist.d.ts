declare module 'mnist' {
  interface MnistSample {
    input: number[];
    output: number[];
  }
  interface MnistSetResult {
    training: MnistSample[];
    test: MnistSample[];
  }
  export function set(training: number, test: number): MnistSetResult;
}
