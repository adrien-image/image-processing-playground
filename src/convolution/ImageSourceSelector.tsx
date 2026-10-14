import { ImageUploader } from '../components/ImageUploader';
import { DEMO_IMAGES, type DemoImage } from './demoImageNames';

const LABELS: Record<DemoImage, string> = {
  checkerboard: 'Checkerboard',
  circle: 'Circle',
  edges: 'Vertical Edge',
  corner: 'Corner',
};

interface ImageSourceSelectorProps {
  active: DemoImage | 'upload';
  onSelect: (name: DemoImage) => void;
  onUpload: (src: string) => void;
  onError: (message: string) => void;
}

export function ImageSourceSelector({ active, onSelect, onUpload, onError }: ImageSourceSelectorProps) {
  return (
    <div className="image-source-selector">
      <h3>Demo Image</h3>
      <p className="section-label">Pick a 32×32 test image, or upload your own (shrunk to at most 64 px):</p>
      <div className="demo-buttons">
        {DEMO_IMAGES.map(name => (
          <button
            key={name}
            type="button"
            className={`btn btn-sm${active === name ? ' btn-primary' : ''}`}
            aria-pressed={active === name}
            onClick={() => onSelect(name)}
          >
            {LABELS[name]}
          </button>
        ))}
        <ImageUploader onImage={onUpload} onError={onError} label="📁 Upload" />
      </div>
    </div>
  );
}
