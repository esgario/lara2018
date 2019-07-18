import React from 'react';
import { Text, TouchableOpacity, StyleSheet } from 'react-native'

const StyledButton = ({ onPress , children, style}) => {
    return (
            <TouchableOpacity 
                style = {[styles.button, style]}
                onPress = {onPress}
            >
                <Text style = {styles.text}>
                    {children}
                </Text>
            </TouchableOpacity>
    );
};

export default StyledButton;


const styles = StyleSheet.create({ 
    button: {
        alignSelf: 'stretch',
        backgroundColor: '#39b500',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#39b500',
        marginHorizontal: 5,
        marginVertical: 20,
        marginTop: 0,
        paddingVertical: 0,
        paddingHorizontal: 0,
    }, 
    text: {
        alignSelf: 'center',
        color: 'white',
        fontSize: 18,
        fontWeight: '600',
        paddingTop: 10,
        paddingBottom: 10
    }
});