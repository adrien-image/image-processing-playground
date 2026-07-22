interface ImageSourceSelectorProps {
  onSelect: (name: 'checkerboard' | 'circle' | 'edges' | 'corner') => void;
  disabled?: boolean;
}

export function ImageSourceSelector({ onSelect, disabled }: ImageSourceSelectorProps) {
  return (
    <div className="image-source-selector">
      <h3>Demo Image</h3>
      <p className="section-label">Pick a 32×32 test image:</p>
      <div className="demo-buttons">
        <button className="btn btn-sm" disabled={disabled} onClick={() => onSelect('checkerboard')}>
          Checkerboard
        </button>
        <button className="btn btn-sm" disabled={disabled} onClick={() => onSelect('circle')}>
          Circle
        </button>
        <button className="btn btn-sm" disabled={disabled} onClick={() => onSelect('edges')}>
          Vertical Edge
        </button>
        <button className="btn btn-sm" disabled={disabled} onClick={() => onSelect('corner')}>
          Corner
        </button>
      </div>
    </div>
  );
}
