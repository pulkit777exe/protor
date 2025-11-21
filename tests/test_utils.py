"""Unit tests for protor.utils module"""
import pytest
import os
import json
from protor.utils import safe_filename, save_json, timestamp, get_default_output_dir


class TestSafeFilename:
    """Tests for safe_filename function"""
    
    def test_basic_alphanumeric(self):
        """Test that alphanumeric characters are preserved"""
        assert safe_filename("test123") == "test123"
    
    def test_special_characters_replaced(self):
        """Test that special characters are replaced with underscores"""
        assert safe_filename("test@#$%file") == "test____file"
    
    def test_spaces_replaced(self):
        """Test that spaces are replaced"""
        assert safe_filename("test file name") == "test_file_name"
    
    def test_url_domain(self):
        """Test converting URL domain to safe filename"""
        assert safe_filename("example.com") == "example_com"
    
    def test_empty_string(self):
        """Test empty string input"""
        assert safe_filename("") == ""
    
    def test_hyphens_preserved(self):
        """Test that hyphens are preserved"""
        assert safe_filename("test-file-name") == "test-file-name"
    
    def test_underscores_preserved(self):
        """Test that underscores are preserved"""
        assert safe_filename("test_file_name") == "test_file_name"


class TestSaveJson:
    """Tests for save_json function"""
    
    def test_save_simple_dict(self, temp_dir):
        """Test saving a simple dictionary"""
        data = {"key": "value", "number": 42}
        path = os.path.join(temp_dir, "test.json")
        
        save_json(data, path)
        
        assert os.path.exists(path)
        with open(path, 'r') as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_creates_directory(self, temp_dir):
        """Test that save_json creates parent directories"""
        data = {"test": "data"}
        path = os.path.join(temp_dir, "subdir", "nested", "test.json")
        
        save_json(data, path)
        
        assert os.path.exists(path)
    
    def test_unicode_content(self, temp_dir):
        """Test saving unicode content"""
        data = {"message": "Hello 世界 🌍"}
        path = os.path.join(temp_dir, "unicode.json")
        
        save_json(data, path)
        
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_nested_structure(self, temp_dir):
        """Test saving nested data structures"""
        data = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"]
                }
            }
        }
        path = os.path.join(temp_dir, "nested.json")
        
        save_json(data, path)
        
        with open(path, 'r') as f:
            loaded = json.load(f)
        assert loaded == data


class TestTimestamp:
    """Tests for timestamp function"""
    
    def test_returns_string(self):
        """Test that timestamp returns a string"""
        result = timestamp()
        assert isinstance(result, str)
    
    def test_format(self):
        """Test timestamp format"""
        result = timestamp()
        # Should be in format: YYYY-MM-DD HH:MM:SS
        parts = result.split()
        assert len(parts) == 2
        date_parts = parts[0].split('-')
        assert len(date_parts) == 3
        time_parts = parts[1].split(':')
        assert len(time_parts) == 3
    
    def test_different_calls(self):
        """Test that consecutive calls produce valid timestamps"""
        ts1 = timestamp()
        ts2 = timestamp()
        # Both should be valid strings
        assert isinstance(ts1, str)
        assert isinstance(ts2, str)


class TestGetDefaultOutputDir:
    """Tests for get_default_output_dir function"""
    
    def test_returns_string(self):
        """Test that function returns a string"""
        result = get_default_output_dir()
        assert isinstance(result, str)
    
    def test_contains_downloads(self):
        """Test that path contains Downloads directory"""
        result = get_default_output_dir()
        assert "Downloads" in result
    
    def test_contains_protor(self):
        """Test that path contains protor subdirectory"""
        result = get_default_output_dir()
        assert "protor" in result
    
    def test_is_absolute_path(self):
        """Test that returned path is absolute"""
        result = get_default_output_dir()
        assert os.path.isabs(result)
