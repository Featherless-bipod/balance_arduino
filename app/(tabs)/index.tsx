import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Fonts } from '@/constants/theme';
import { useThemeColor } from '@/hooks/use-theme-color';
import React, { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

export default function TabThreeScreen() {
  const tint = useThemeColor({}, 'tint');
  const colorL = useThemeColor({}, 'colorL');
  const colorR = useThemeColor({}, 'colorR');

  // functionality for Timer
  const INITIAL_SECONDS = 1 * 60; // 5 minutes
  const [timeLeft, setTimeLeft] = useState<number>(INITIAL_SECONDS);
  const [buttonLabel, setButtonLabel] = useState<string>('Start');
  const intervalRef = useRef<number | null>(null);

  // start or reset depending on current label
  const startTimer = () => {
    if (buttonLabel === 'Start') {
      if (intervalRef.current == null) {
        intervalRef.current = setInterval(() => {
          setTimeLeft(prev => {
            if (prev <= 1) {
              // stop at zero
              if (intervalRef.current != null) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
              }
              setButtonLabel('Start');
              return 0;
            }
            return prev - 1;
          });
        }, 1000) as unknown as number;
      }
    } else {
      // Reset: stop interval and restore initial value
      if (intervalRef.current != null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setTimeLeft(INITIAL_SECONDS);
    }
  };

  const updateButton = () => setButtonLabel(prev => (prev === 'Start' ? 'Reset' : 'Start'));

  const handlePress = () => {
    startTimer();
    updateButton();
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current != null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  function formatTime(sec: number) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  // Functionality for Connect button
  const [connectLabel, setConnectLabel] = useState<string>('Not Connected');
  const updateConnect = () => setConnectLabel(prev => (prev === 'Not Connected' ? 'Success!' : 'Not Connected'));

  const connectDevice = () => {
    updateConnect();
  }

  return (
    <ThemedView style={styles.mainContainer}>
      <ThemedView style={[styles.titleContainer, { backgroundColor: tint }]}> 
        <ThemedText type="title" style={{ fontWeight: 'bold' }}>Balance Exercise</ThemedText>
      </ThemedView>

      <ThemedView style={{ height: 0 }} />

      <ThemedView style={[styles.colContainer, { marginTop: 10, paddingBottom: 10, flex: 1, minHeight: 40 }]}>
        <ThemedView style={[styles.column, { marginRight: 10, flex: 3, justifyContent: 'center' }]}> 
          <ThemedText type="subtitle">Connect device:</ThemedText>
          <ThemedText>{connectLabel}</ThemedText>
        </ThemedView>

        <ThemedView style={[styles.column, { marginLeft: 10, flex: 2, justifyContent: 'center', alignItems: 'flex-end' }]}> 
          <Pressable onPress={connectDevice}>
            <ThemedView style={{ minWidth: 125, backgroundColor: tint, padding: 10, borderRadius: 10}}>
              <ThemedText type="subtitle" style={{ textAlign: 'center' }}>Connect</ThemedText>
            </ThemedView>
          </Pressable>
        </ThemedView>
      </ThemedView>

      <ThemedView style={[styles.colContainer, { flex: 6 }]}> 
        <ThemedView style={[styles.column, { marginRight: 10, flex: 1 }]}> 
          <View style={[styles.body, { backgroundColor: colorL , borderRadius: 10, flex: 7, justifyContent: 'center', alignItems: 'center' }]}> 
            <Text style={{ color: 'white', fontWeight: 'bold', fontSize: 64, lineHeight: 70, fontFamily: Fonts.rounded }}>L</Text>
          </View>
        </ThemedView>

        <ThemedView style={[styles.column, { marginLeft: 10, flex: 1 }]}> 
          <View style={[styles.body, { backgroundColor: colorR , borderRadius: 10, flex: 7, justifyContent: 'center', alignItems: 'center' }]}> 
            <Text style={{ color: 'white', fontWeight: 'bold', fontSize: 64, lineHeight: 70, fontFamily: Fonts.rounded }}>R</Text>
          </View>
        </ThemedView>
      </ThemedView>

      <ThemedView style={[styles.colContainer, { padding: 10, flex: 1, justifyContent: 'center', alignContent: 'center' }]}> 
        <ThemedView style={{ paddingRight: 10, justifyContent: 'center', alignContent: 'center' }}> 
          <ThemedText type="title">Timer:</ThemedText>
        </ThemedView>

        <ThemedView style={{ paddingLeft: 10, paddingRight: 10, justifyContent: 'center', alignContent: 'center' }}> 
          <ThemedText type="title" style={{ fontSize: 42, lineHeight: 40 }}>{formatTime(timeLeft)}</ThemedText>
        </ThemedView>
      </ThemedView>

      <ThemedView style={{ paddingBottom: 20, justifyContent: 'center', alignItems: 'center' }}>
        <Pressable onPress={handlePress}>
          <ThemedView style={{ backgroundColor: tint, paddingHorizontal: 20, paddingVertical: 10, width: 150 , borderRadius: 10}}>
            <ThemedText type="title" style={{ textAlign: 'center' }}>{buttonLabel}</ThemedText>
          </ThemedView>
        </Pressable>
      </ThemedView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    backgroundColor: '#A0C1D6',
    padding: 20,
    flexDirection: 'row',
    gap: 8,
    height: '20%',
    // align children to the bottom of this container
    alignItems: 'flex-end',
    // keep horizontal flow starting at left
    justifyContent: 'flex-start',
  },

  mainContainer: {
    display: 'flex',
    flex: 1,
    flexDirection: 'column',
  },

  colContainer: {
    paddingLeft: 30,
    paddingRight: 30,
    display: 'flex',
    flex: 1,
    flexDirection: 'row',
  },

  body: {
    padding: 20,
    flex: 1,
    display: 'flex',
  },

  column: {
    display: 'flex',
    flex: 1,
  },

  label: {
    height: 50,
  },
});
