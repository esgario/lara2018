import React, { Component } from 'react';

import { 
        StyleSheet,
        Text,
        View,
        Platform, 
        Image, 
        TouchableOpacity, 
        Dimensions } from 'react-native';

import Icon from 'react-native-vector-icons/FontAwesome';

import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import * as Permissions from 'expo-permissions';

const screenWidth = Math.round(Dimensions.get('window').width);
const screenHeight = Math.round(Dimensions.get('window').height);

class Menu extends Component {

    static navigationOptions = {

        title: 'Menu',
        headerStyle: {
          backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: 30
        },

    };

    state = {

        nomeUsuarioLogado: '',
        image: null,
        latitude: null,
        longitude: null,
        hasCameraPermission: null,
        hasCameraAlbumPermission: null,
        hasLocationPermission: null

    };

    /**
     * Método para pegar os dados da tela anterior ao montar a tela
     * @author Pedro Biasutti 
     * */
    componentDidMount () {

        const { navigation } = this.props;
        const nomeUsuario = navigation.getParam('nomeUsuario', 'erro nomeUsuario');

        this.setState({nomeUsuarioLogado: nomeUsuario});

    };

    /**
     * Método para tirar foto via câmera do celular
     * @author Pedro Biasutti
     */
    takeimage = async () => {

        const { status } = await Permissions.askAsync(Permissions.CAMERA);
        this.setState({ hasCameraPermission: status === 'granted' });

        const permission = await Permissions.askAsync(Permissions.CAMERA);
        
        if (permission.status !== 'granted') {

        } else {

            await this.pegaCoordenadas();
            
            let result = await ImagePicker.launchCameraAsync({
                allowsEditing: Platform.OS === 'ios' ? false : true,
                // allowsEditing: true,
                // aspect: [3, 3],
            });
        
            if (!result.cancelled) {
                this.setState({ image: result });
            }
        }

        if (this.state.image !== null) {

            let image = this.state.image;

            this.setState({image: null});
            this.props.navigation.navigate('Aquisicao',{ 
                nomeUsuario: this.state.nomeUsuarioLogado,
                image: image,
                latitude: this.state.latitude,
                longitude: this.state.longitude
            });
                
        }

    };

    /**
     * Método para escolher uma imagem do álbum de fotos
     * @author Pedro Biasutti
     */
    pickImage = async () => {

        const { status } = await Permissions.askAsync(Permissions.CAMERA_ROLL);
        this.setState({ hasCameraAlbumPermission: status === 'granted' });

        const permission = await Permissions.askAsync(Permissions.CAMERA_ROLL);

        if (permission.status !== 'granted') {

        } else {

            await this.pegaCoordenadas();

            let result = await ImagePicker.launchImageLibraryAsync({
                allowsEditing: Platform.OS === 'ios' ? false : true,
                // allowsEditing: true,
                // aspect: [5, 5],
            });
        
            if (!result.cancelled) {
                this.setState({ image: result });
            }   

        }

        if (this.state.image !== null) {

            let image = this.state.image;

            this.setState({image: null});
            this.props.navigation.navigate('Aquisicao',{ 
                nomeUsuario: this.state.nomeUsuarioLogado,
                image: image,
                latitude: this.state.latitude,
                longitude: this.state.longitude
            });

        }

    };

    /**
     * Método para pegar a localização do usuário e salva-la
     * @author Pedro Biasutti
     */
    pegaCoordenadas = async () => {

        const { status } = await Permissions.askAsync(Permissions.LOCATION);
        this.setState({ hasLocationPermission: status === 'granted' });

        const permission = await Permissions.getAsync(Permissions.LOCATION);

        if (permission.status !== 'granted') {

            this.setState({latitude: null, longitude: null})

            console.log('DEU ERRO PEGA COORDENADAS');

        }   else {

            let location = await Location.getCurrentPositionAsync({});

            console.log('NÃO DEU ERRO PEGA COORDENADAS');

            this.setState({latitude: location.coords.latitude, longitude: location.coords.longitude})

        }     

    };

    render () {

        return (

            <View style = {styles.menuContainer}>

                <View style = {{marginBottom: -40, marginTop: -20}}>

                    <Image
                        style={{width: 0.5 * screenWidth, height: 0.3 * screenHeight}}
                        source = {require('./../assets/logo02_sb.png')}
                        resizeMode = 'contain'
                    />

                </View>

                <View style = {styles.topButtonsContainer}>

                    <TouchableOpacity 
                        style = {styles.circularButton}
                        onPress = {() => this.takeimage()}
                    >
                        <Icon name = 'camera' size={42} color="#2b2c2d"/>
                    </TouchableOpacity>

                    <Text style = {[styles.hiperlink, {marginBottom: 20}]}
                        onPress={() => this.pickImage()}
                    >
                      Selecione do álbum
                    </Text>

                </View>

                <View style = {styles.centerButtonsContainer}>

                    <TouchableOpacity 
                        style = {styles.button}
                        onPress = {() => this.props.navigation.navigate('Biblioteca', {nomeUsuario: this.state.nomeUsuarioLogado})}                         
                    >
                        <Icon  flex = {1} paddingLeft = {20} paddingRight = {20} name = 'list-ul' size={40} color="white"/>
                        <Text style = {styles.text}>Biblioteca</Text>
                    </TouchableOpacity>

                    <TouchableOpacity 
                        style = {styles.button}
                        onPress = {() => this.props.navigation.navigate('Sobre')}
                    >
                        <Icon  flex = {1} paddingLeft = {20} paddingRight = {20} name = 'exclamation-circle' size={40} color="white"/>
                        <Text style = {styles.text}>Sobre</Text>
                    </TouchableOpacity>

                </View>

                <View style = {styles.endButtonsContainer}>

                    <TouchableOpacity 
                        style={styles.sideButton}
                        onPress = {() => this.props.navigation.navigate('DadosCadastrais', {nomeUsuario: this.state.nomeUsuarioLogado})}
                    >
                        <Icon name = 'user' size={30} color="#2b2c2d"/>
                    </TouchableOpacity> 

                    <TouchableOpacity 
                        style={styles.sideButton}
                        onPress = {() => this.props.navigation.navigate('LogOut')}
                    >   
                        <Icon name = 'sign-out' size={30} color="#2b2c2d"/>
                    </TouchableOpacity>  

                </View>

            </View>
            
        );

    };

};

export default Menu;

const fontStyle = Platform.OS === 'ios' ? 'Arial Hebrew' : 'serif';

const styles = StyleSheet.create({

    headers: { 
      fontFamily: fontStyle,
      color: '#33ccff',
      fontWeight: 'bold',
      fontSize: 28,
      marginTop: 20,
    }, 
    button: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        width: '60%',
        backgroundColor: '#39b500',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#39b500',
        paddingVertical: 5,
        paddingHorizontal: 10,
        marginHorizontal: 5,
        marginBottom: 10,
    },
    circularButton: {
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#e2e2e2',
        height: screenWidth * 0.45,
        width: screenWidth * 0.45,
        borderRadius: screenWidth * 0.45,
        borderWidth: 1,
        borderColor: '#d2d2d2',
        marginHorizontal: 5,

    },
    sideButton: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        width: '60%',
        backgroundColor: '#e2e2e2',
        borderRadius: 1,
        borderWidth: 1,
        borderColor: '#d2d2d2',
        paddingVertical: 5,
    },
    text: {
        flex: 2,
        textAlign: 'center',
        alignSelf: 'center',
        color: 'white',
        fontSize: 20,
        fontWeight: '600',
        paddingVertical: 10,
        paddingHorizontal: 20,
    },
    menuContainer: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    topButtonsContainer: {
        marginVertical: 10,
        alignItems: 'center'
    },
    centerButtonsContainer: {
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '25%'
    },
    endButtonsContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center'
    },
    hiperlink: {
        alignSelf: 'center',
        fontSize: 20,
        textDecorationLine: 'underline', 
        color: '#39b500',
        marginTop: 10 
    },

});