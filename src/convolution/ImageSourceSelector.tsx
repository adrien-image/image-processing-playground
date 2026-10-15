import { ImageUploader } from '../components/ImageUploader';
import { DEMO_IMAGES, type DemoImage } from './demoImageNames';

export type Resolution = 'small' | 'large';

const LABELS: Record<DemoImage, string> = {
  scene: 'Scene',
  checkerboard: 'Checkerboard',
  circle: 'Circle',
  edges: 'Vertical Edge',
  corner: 'Corner',
};

interface ImageSourceSelectorProps {
  active: DemoImage | 'upload';
  resolution: Resolution;
  onSelect: (name: DemoImage) => void;
  onUpload: (src: string) => void;
  onResolutionChange: (resolution: Resolution) => void;
  onError: (message: string) => void;
}

export function ImageSourceSelector({ active, resolution, onSelect, onUpload, onResolutionChange, onError }: ImageSourceSelectorProps) {
  return (
    <div className="image-source-selector">
      <h3>Demo Image</h3>
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
        <ImageUploader
          onImage={onUpload}
          onError={onError}
          label={active === 'upload' ? '📁 Upload another' : '📁 Upload'}
        />
      </div>

      <p className="section-label" id="conv-resolution-label">Resolution</p>
      <div className="demo-buttons" role="group" aria-labelledby="conv-resolution-label">
        <button
          type="button"
          className={`btn btn-sm${resolution === 'large' ? ' btn-primary' : ''}`}
          aria-pressed={resolution === 'large'}
          onClick={() => onResolutionChange('large')}
        >
          Large · 256 px
        </button>
        <button
          type="button"
          className={`btn btn-sm${resolution === 'small' ? ' btn-primary' : ''}`}
          aria-pressed={resolution === 'small'}
          onClick={() => onResolutionChange('small')}
        >
          Small · 32 px
        </button>
      </div>
      <p className="resolution-note">
        {resolution === 'large'
          ? 'See the effect on a real-sized picture. Uploads are shrunk to at most 256 px.'
          : 'Few enough pixels to follow every multiply-and-sum. Uploads are shrunk to at most 64 px.'}
      </p>
    </div>
  );
}
