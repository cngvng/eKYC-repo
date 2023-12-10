// ignore_for_file: use_build_context_synchronously

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _result = '';

  Future<void> _captureAndSendImage() async {
    final imagePicker = ImagePicker();
    final image = await imagePicker.pickImage(source: ImageSource.camera);

    if (image == null) {
      setState(() {
        _result = 'Failed to capture image.';
      });
      return;
    }

    final bytes = await image.readAsBytes();

    final response = await http.post(
      Uri.parse('http://localhost:8000/upload'),
      body: bytes,
      headers: {'Content-Type': 'application/octet-stream'},
    );

    if (response.statusCode == 200) {
      final result = response.body;
      setState(() {
        _result = result;
      });

      // Hiển thị kết quả trong hộp thoại
      showDialog(
        context: context,
        builder: (context) {
          return AlertDialog(
            title: const Text('Result'),
            content: Text(_result),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          );
        },
      );
    } else {
      setState(() {
        _result = 'Failed to send image.';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Image Capture'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () => _captureAndSendImage(),
              child: const Text('Capture Image'),
            ),
          ],
        ),
      ),
    );
  }
}