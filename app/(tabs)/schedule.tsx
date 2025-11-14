import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Fonts } from '@/constants/theme';
import { useThemeColor } from '@/hooks/use-theme-color';
import { StyleSheet, Text, View } from 'react-native';

//import MySearchIcon from '@/components/icons/foot-print';

export default function TabThreeScreen() {
  const titleBg = useThemeColor({}, 'tint');
  const colorL = useThemeColor({}, 'colorL');
  const colorR = useThemeColor({}, 'colorR');
  
  return (
    <ThemedView style={styles.mainContainer}>
      
      
      <ThemedView style={[styles.titleContainer, {backgroundColor: titleBg}]}>
        <ThemedText type="title" style={{fontFamily: Fonts.rounded,}}>Balance Exercise</ThemedText>
      </ThemedView>

      <ThemedView style={{ height: 0 }}>
        
      </ThemedView>

      <ThemedView style={[styles.colContainer, {marginTop: 20, flex: 1, minHeight: 40}]}>
        <ThemedView style={[styles.column, {marginRight: 10, flex: 1}]}>
          <ThemedText>Turn </ThemedText>
          </ThemedView>
        <ThemedView style={[styles.column, {marginLeft: 10, flex: 1}]}>
          <ThemedText>Connect</ThemedText>
          </ThemedView>
      </ThemedView>

      <ThemedView style={[styles.colContainer, {flex: 6}]} >
        <ThemedView style={[styles.column, {marginRight: 10, flex: 1}]}>
              
          <View style={[styles.body, {backgroundColor: colorL, flex: 7, justifyContent: 'center', alignItems: 'center' }]}>
            <Text style={{color: 'white', fontWeight: 'bold', fontSize: 64, lineHeight: 70, fontFamily: Fonts.rounded}}>L</Text>
          </View>
        </ThemedView>

        <ThemedView style={[styles.column, {marginLeft: 10, flex: 1}]}>
              
          <View style={[styles.body, {backgroundColor: colorR, flex: 7, justifyContent: 'center', alignItems: 'center' }]}>
            
            <Text style={{color: 'white', fontWeight: 'bold', fontSize: 64, lineHeight: 70, fontFamily: Fonts.rounded}}>R</Text>
          </View>
        </ThemedView>
      </ThemedView>

      <ThemedView style={[styles.body, {flex: 2, justifyContent: "center"}]}>
        <ThemedText type="subtitle" style={{textAlign: "center"}}>Timer:</ThemedText>
        <ThemedText style={{fontSize: 20}}/>
        <ThemedText type="title" style={{fontSize: 32, textAlign: "center"}}>5:00</ThemedText>
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
    height: "20%",
    // align children to the bottom of this container
    alignItems: 'flex-end',
    // keep horizontal flow starting at left
    justifyContent: "flex-start",
    //alignContent: "center",
  },

  mainContainer: {
    display: 'flex',
    flex: 1,
    flexDirection: 'column',
  },

  colContainer: {
    paddingLeft: 30,
    paddingRight: 30,
    display: "flex",
    flex: 1,
    flexDirection: "row"
  },

  body: {
    //backgroundColor: "white",
    padding: 20,
    flex: 1,
    display: 'flex',
  },

  column: {
    //backgroundColor: "grey",
    //padding: 10,
    //margin: 10,
    display: "flex",
    flex: 1,
  },

  label: {
    height: 50,
    
  }
});
