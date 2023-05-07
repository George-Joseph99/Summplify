import React, { useState, useEffect } from 'react' ;
import axios from "axios";
import { GoCloudUpload } from "react-icons/go";
import { AiTwotoneFileAdd } from "react-icons/ai";
import { AiTwotoneDelete } from "react-icons/ai";
import './Output.css'
import { List, Card, Button } from 'antd';
import { useSearchParams } from 'react-router-dom';
import Search from 'antd/es/transfer/search';

function Output() {
    const [searchparams] = useSearchParams();
    console.log(searchparams.get("text"))
    // const text = navigate.state?.data || '';

    const [fileUpload, setFileUpload] = useState(false);
    const [fileName, setFileName] = useState('');
    const [file, setFile] = useState('');
    // const [text, setText] = useState('');
    var files = [];

    // useEffect(() => {
    //     axios.get('http://localhost:3000/output') // 1 will be changed to id
    //     .then((response) => {
    //     const data = response.data;
    //     setText(data['text']);

    //     }).catch(() => {
    //         alert('Error retrieving data');
    //     })
    //     }, []);

    // const handleTextInput = (e) => {
    //     console.log(e.target.value);
    //     setText(e.target.value);
    // }


    const handleDeleteIconClick = () => {
        setFileName('');
        setFileUpload(false);
        setFile('');
    }

  return (
    <div className="div_container">
      <p className="p_output">{searchparams.get("text")}</p>
    </div>
  );
}

export default Output;
