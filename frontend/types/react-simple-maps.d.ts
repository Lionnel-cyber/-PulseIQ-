// Minimal type shim for react-simple-maps@3 (does not ship .d.ts files).
// Covers only what WorldMap.tsx uses.

declare module "react-simple-maps" {
  import type { ReactNode, CSSProperties } from "react";

  interface ComposableMapProps {
    projection?: string;
    style?: CSSProperties;
    children?: ReactNode;
  }
  export function ComposableMap(props: ComposableMapProps): JSX.Element;

  interface GeographiesProps {
    geography: string;
    children: (props: { geographies: Geography[] }) => ReactNode;
  }
  export function Geographies(props: GeographiesProps): JSX.Element;

  interface Geography {
    rsmKey: string;
    [key: string]: unknown;
  }
  interface GeographyProps {
    geography: Geography;
    style?: {
      default?: CSSProperties;
      hover?: CSSProperties;
      pressed?: CSSProperties;
    };
  }
  export function Geography(props: GeographyProps): JSX.Element;

  interface MarkerProps {
    coordinates: [number, number];
    key?: string;
    children?: ReactNode;
  }
  export function Marker(props: MarkerProps): JSX.Element;
}
