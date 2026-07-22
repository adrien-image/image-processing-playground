import { useRef, useCallback } from 'react';

interface ImageUploaderProps {
  onImage: (img: HTMLImageElement) => void;
  onError?: (error: string) => void;
  label?: string;
  disabled?: boolean;
}

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB
const MAX_DIMENSION = 8192;              // cap each dimension at 8192 px

export function ImageUploader({ onImage, onError, label = '📁 Upload image', disabled }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reject oversized files before reading
    if (file.size > MAX_FILE_SIZE) {
      onError?.(`File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB (max ${MAX_FILE_SIZE / 1024 / 1024} MB)`);
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        // Clamp dimensions to avoid DoS from massive images
        if (img.naturalWidth > MAX_DIMENSION || img.naturalHeight > MAX_DIMENSION) {
          const ratio = Math.min(MAX_DIMENSION / img.naturalWidth, MAX_DIMENSION / img.naturalHeight);
          const w = Math.round(img.naturalWidth * ratio);
          const h = Math.round(img.naturalHeight * ratio);
          const canvas = document.createElement('canvas');
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext('2d')!;
          ctx.drawImage(img, 0, 0, w, h);
          const resized = new Image(w, h);
          resized.src = canvas.toDataURL();
          resized.onload = () => onImage(resized);
          return;
        }
        onImage(img);
      };
      img.onerror = () => onError?.('Failed to decode image');
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  }, [onImage, onError]);

  return (
    <>
      <input ref={inputRef} type="file" accept="image/*" onChange={handleFile} style={{ display: 'none' }} />
      <button className="btn btn-sm" onClick={() => inputRef.current?.click()} disabled={disabled}>
        {label}
      </button>
    </>
  );
}
