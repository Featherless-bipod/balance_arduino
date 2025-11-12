import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Fonts } from '@/constants/theme';
import { StyleSheet, View } from 'react-native';




//import MySearchIcon from '@/components/icons/foot-print';

export default function TabThreeScreen() {
  return (
    <ThemedView style={styles.mainContainer}>
      
      
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title" style={{fontFamily: Fonts.rounded,}}>Balance Exercise</ThemedText>
      </ThemedView>

      <ThemedView style={[styles.body, {flex: 1, minHeight: 40}]}>
        <ThemedText>Turn on your Balance Buzz pad and perform your balance exercise.</ThemedText>
      </ThemedView>

      <ThemedView style={[styles.colContainer, {flex: 6}]} >
        <ThemedView style={[styles.column, {marginRight: 10, flex: 1}]}>
          <View style={{height: 50, justifyContent: "center"}}>
            <ThemedText type={"subtitle"} style={{textAlign: "center"}}>Left</ThemedText>
          </View>
    
          <View style={[styles.body, {backgroundColor: "#d83c3cff", flex: 7}]}>
            <ThemedText> </ThemedText> 
          </View>
        </ThemedView>

        <ThemedView style={[styles.column, {marginLeft: 10, flex: 1}]}>
          <View style={{height: 50, justifyContent: "center"}}>
            <ThemedText type={"subtitle"} style={{textAlign: "center"}}>Right</ThemedText>
          </View>
    
          <View style={[styles.body, {backgroundColor: "#3939e8ff", flex: 7}]}>
            <ThemedText> </ThemedText>
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
  headerImage: {
    color: '#808080',
    bottom: -90,
    left: -35,
    position: 'absolute',
  },
  titleContainer: {
    backgroundColor: '#6da6dcff',
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
