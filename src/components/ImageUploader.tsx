import { useEffect, useRef, type ChangeEvent } from 'react';

interface ImageUploaderProps {
  /** Receives an object URL for the chosen file (valid until the next upload or unmount). */
  onImage: (src: string) => void;
  onError?: (error: string) => void;
  label?: string;
  disabled?: boolean;
}

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

/**
 * File picker for images. Decoding, the pixel-count limit and downscaling happen in
 * `useImageData`, which reports decode failures through its `error` value.
 */
export function ImageUploader({ onImage, onError, label = '📁 Upload image', disabled }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const urlRef = useRef<string | null>(null);

  // Release the last object URL when the uploader goes away.
  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
  }, []);

  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const input = e.target;
    const file = input.files?.[0];
    // Reset so choosing the same file again still fires `change`.
    input.value = '';
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      onError?.(`"${file.name}" is not an image.`);
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      onError?.(`File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB (max ${MAX_FILE_SIZE / 1024 / 1024} MB)`);
      return;
    }

    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = URL.createObjectURL(file);
    onImage(urlRef.current);
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleFile}
        hidden
        aria-hidden="true"
        tabIndex={-1}
      />
      <button type="button" className="btn btn-sm" onClick={() => inputRef.current?.click()} disabled={disabled}>
        {label}
      </button>
    </>
  );
}
