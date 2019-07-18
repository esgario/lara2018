import React, { Component } from 'react';
import {
        View, 
    } from 'react-native';




class LogOut extends Component {

    static navigationOptions = {
        title: 'Log out',
        headerStyle: {
          backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: 30
        },
    };


    render () {
        return (
            <View></View>
        );
    }
}

export default LogOut;

