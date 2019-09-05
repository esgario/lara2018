import React, { Component } from 'react';
import { createStackNavigator, createAppContainer } from 'react-navigation';

import Home from './Screen/Home';
import Cadastro from './Screen/Cadastro';
import DadosCadastrais from './Screen/DadosCadastrais';
import RecuperarSenha from './Screen/RecuperarSenha';
import Menu from './Screen/Menu';
import Aquisicao from './Screen/Aquisicao';
import LogOut from './Screen/LogOut';
import Sobre from './Screen/Sobre';
import Biblioteca from './Screen/Biblioteca';
import Resultado from './Screen/Resultado';
import Tratamento from './Screen/Tratamento';
import Visualiza from './Screen/Visualiza';


export default class App extends Component {

  render() {
    return (
        <AppContainer/>
    );
  }
}

const AppStackNavigator = createStackNavigator({
  Home: Home,
  Cadastro: Cadastro,
  DadosCadastrais: DadosCadastrais,
  RecuperarSenha: RecuperarSenha,
  Menu: Menu,
  Aquisicao: Aquisicao,
  LogOut: LogOut,
  Sobre: Sobre,
  Biblioteca: Biblioteca,
  Resultado: Resultado,
  Tratamento: Tratamento,
  Visualiza: Visualiza
  },
  {
    initialRouteName: 'Home'
})

const AppContainer = createAppContainer(AppStackNavigator);

