import React, { Component } from 'react';
import {
        View,
        ActivityIndicator,
        StyleSheet,
        AsyncStorage 
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

    /**
     * Método para apagar da memória as variáveis de log in assim que montar a tela
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        // Apaga as variáveis
        await this._removeData('nomeUsuario');
        await this._removeData('senha');

        // Volta para o Home
        this.props.navigation.navigate('Home')
        
    };

    /**
     * Método para remover variável salva na memória do celular
     * @author Pedro Biasutti
     * @param storage_Key - chave de acesso (nome salvo)
     */
    _removeData = async (storage_Key) => {

        try {

            await AsyncStorage.removeItem(storage_Key);

            console.log('NÃO DEU ERRO REMOVE DA MEMÓRIA');

        } catch (error) {

            console.log('DEU ERRO REMOVE DA MEMÓRIA');

        }

    };


    render () {

        return (

            <View style = {styles.activity}>

                <ActivityIndicator/>

            </View>
        );

    };

}

export default LogOut;

const styles = StyleSheet.create({
    
    activity : {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        transform: ([{ scaleX: 2.5 }, { scaleY: 2.5 }]),
    },

});

