import type { Processor } from './types';

const processors = new Map<string, Processor>();

export function registerProcessor(processor: Processor): void {
  if (processors.has(processor.id)) {
    console.warn(`Processor "${processor.id}" is already registered — overwriting.`);
  }
  processors.set(processor.id, processor);
}

export function getProcessor(id: string): Processor | undefined {
  return processors.get(id);
}

export function getAllProcessors(): Processor[] {
  return Array.from(processors.values());
}

export function getProcessorIds(): string[] {
  return Array.from(processors.keys());
}
