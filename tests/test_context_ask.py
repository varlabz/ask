import pytest
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from ask.core.agent_context import ContextASK


class TestContextASK:
    """Test suite for ContextASK base class."""

    def test_to_input_basic_fields(self):
        """Test to_input() with basic field types."""
        class TestModel(ContextASK):
            name: str = Field(description="The name field")
            age: int
            height: float
            is_active: bool

        expected = "<name>The name field</name>\n<age>number</age>\n<height>number</height>\n<is_active>boolean</is_active>"
        result = TestModel.to_input()
        assert result == expected

    def test_to_input_no_descriptions(self):
        """Test to_input() with fields that have no descriptions."""
        class TestModel(ContextASK):
            name: str
            count: int

        expected = "<name>string</name>\n<count>number</count>"
        result = TestModel.to_input()
        assert result == expected

    def test_to_input_mixed_descriptions(self):
        """Test to_input() with some fields having descriptions and others not."""
        class TestModel(ContextASK):
            title: str = Field(description="The title")
            count: int
            active: bool = Field(description="Is active status")

        expected = "<title>The title</title>\n<count>number</count>\n<active>Is active status</active>"
        result = TestModel.to_input()
        assert result == expected

    def test_to_input_empty_model(self):
        """Test to_input() with a model that has no fields."""
        class EmptyModel(ContextASK):
            pass

        result = EmptyModel.to_input()
        assert result == ""

    def test_to_output_basic_values(self):
        """Test to_output() with basic field values."""
        class TestModel(ContextASK):
            name: str = Field(description="The name field")
            age: int
            height: float
            is_active: bool

        instance = TestModel(name="John", age=25, height=5.9, is_active=True)
        expected = "<name>John</name>\n<age>25</age>\n<height>5.9</height>\n<is_active>True</is_active>"
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_none_values(self):
        """Test to_output() with None values."""
        class TestModel(ContextASK):
            name: str
            optional_field: Optional[str] = None

        instance = TestModel(name="Test", optional_field=None)
        expected = "<name>Test</name>\n"
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_list(self):
        """Test to_output() with list field."""
        class TestModel(ContextASK):
            tags: list[str]

        instance = TestModel(tags=["python", "test", "mcp"])
        expected = "<tags>python</tags><tags>test</tags><tags>mcp</tags>"
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_empty_list(self):
        """Test to_output() with empty list."""
        class TestModel(ContextASK):
            tags: list[str]

        instance = TestModel(tags=[])
        expected = ""
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_dict(self):
        """Test to_output() with dictionary field."""
        class TestModel(ContextASK):
            metadata: dict[str, str]

        instance = TestModel(metadata={"key1": "value1", "key2": "value2"})
        # Dictionary values are JSON-encoded
        expected = '<metadata>{"key1": "value1", "key2": "value2"}</metadata>'
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_enum(self):
        """Test to_output() with enum field."""
        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class TestModel(ContextASK):
            status: Status

        instance = TestModel(status=Status.ACTIVE)
        expected = "<status>active</status>"
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_nested_model(self):
        """Test to_output() with nested BaseModel."""
        class Address(BaseModel):
            street: str
            city: str

        class Person(ContextASK):
            name: str
            address: Address

        address = Address(street="123 Main St", city="Anytown")
        person = Person(name="John Doe", address=address)
        expected = "<name>John Doe</name>\n<street>123 Main St</street>\n<city>Anytown</city>"
        result = person.to_output()
        assert result == expected

    def test_to_output_complex_nested(self):
        """Test to_output() with complex nested structures."""
        class Item(BaseModel):
            name: str
            quantity: int

        class Order(ContextASK):
            order_id: str
            items: list[Item]
            metadata: dict[str, str]

        order = Order(
            order_id="ORD-001",
            items=[
                Item(name="Widget A", quantity=5),
                Item(name="Widget B", quantity=3)
            ],
            metadata={"priority": "high", "source": "web"}
        )

        result = order.to_output()
        # Check that all parts are present
        assert "<order_id>ORD-001</order_id>" in result
        assert "<name>Widget A</name>" in result
        assert "<quantity>5</quantity>" in result
        assert "<name>Widget B</name>" in result
        assert "<quantity>3</quantity>" in result
        assert '"priority": "high"' in result
        assert '"source": "web"' in result

    def test_to_output_special_characters(self):
        """Test to_output() with special characters that need XML escaping."""
        class TestModel(ContextASK):
            content: str

        instance = TestModel(content='Special chars: <>&"\'')
        expected = '<content>Special chars: &lt;&gt;&amp;"\'</content>'
        result = instance.to_output()
        assert result == expected

    def test_to_output_numeric_types(self):
        """Test to_output() with various numeric types."""
        class TestModel(ContextASK):
            integer_field: int
            float_field: float

        instance = TestModel(integer_field=42, float_field=3.14159)
        result = instance.to_output()
        assert "<integer_field>42</integer_field>" in result
        assert "<float_field>3.14159</float_field>" in result

    def test_to_output_boolean_values(self):
        """Test to_output() with boolean values."""
        class TestModel(ContextASK):
            flag_true: bool
            flag_false: bool

        instance = TestModel(flag_true=True, flag_false=False)
        result = instance.to_output()
        assert "<flag_true>True</flag_true>" in result
        assert "<flag_false>False</flag_false>" in result

    def test_inheritance_works(self):
        """Test that ContextASK methods work with inheritance."""
        class BaseModel(ContextASK):
            base_field: str = Field(description="Base field")

        class DerivedModel(BaseModel):
            derived_field: int

        # Test class method
        result = DerivedModel.to_input()
        assert "<base_field>Base field</base_field>" in result
        assert "<derived_field>number</derived_field>" in result

        # Test instance method
        instance = DerivedModel(base_field="test", derived_field=123)
        result = instance.to_output()
        assert "<base_field>test</base_field>" in result
        assert "<derived_field>123</derived_field>" in result

    def test_multiple_instances_independence(self):
        """Test that multiple instances produce independent output."""
        class TestModel(ContextASK):
            value: str

        instance1 = TestModel(value="first")
        instance2 = TestModel(value="second")

        result1 = instance1.to_output()
        result2 = instance2.to_output()

        assert "<value>first</value>" in result1
        assert "<value>second</value>" in result2
        assert "first" not in result2
        assert "second" not in result1

    # Negative tests

    def test_to_output_missing_required_field_raises_error(self):
        """Test that missing required fields raise validation error."""
        class TestModel(ContextASK):
            required_field: str  # No default value

        with pytest.raises(ValidationError):
            TestModel()  # Missing required_field

    def test_to_output_invalid_type_raises_error(self):
        """Test that invalid field types raise validation error."""
        class TestModel(ContextASK):
            number_field: int

        with pytest.raises(ValidationError):
            TestModel(number_field="not_a_number")

    def test_to_input_unknown_field_type(self):
        """Test to_input() with unknown field type annotation."""
        class TestModel(ContextASK):
            custom_field: object  # Unknown type

        result = TestModel.to_input()
        assert "<custom_field>unknown</custom_field>" in result

    def test_to_output_none_model_instance(self):
        """Test to_output() behavior with None values in complex fields."""
        class TestModel(ContextASK):
            optional_model: Optional[BaseModel] = None

        instance = TestModel(optional_model=None)
        result = instance.to_output()
        assert result == ""

    def test_to_output_empty_strings(self):
        """Test to_output() with empty string values."""
        class TestModel(ContextASK):
            empty_str: str
            whitespace_str: str

        instance = TestModel(empty_str="", whitespace_str="   ")
        result = instance.to_output()
        assert "<empty_str></empty_str>" in result
        assert "<whitespace_str>   </whitespace_str>" in result