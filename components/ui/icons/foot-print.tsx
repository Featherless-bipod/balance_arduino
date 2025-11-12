// components/icons/MySearchIcon.tsx
import React from 'react';
import { OpaqueColorValue } from 'react-native';
import Svg, { Path } from 'react-native-svg';

export default function MySearchIcon({
  size = 24,
  color = '#000',
}: {
  size?: number;
  color?: string | OpaqueColorValue;
}) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Path d="M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z" stroke={String(color)} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      <Path d="M21 21l-4.35-4.35" stroke={String(color)} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}