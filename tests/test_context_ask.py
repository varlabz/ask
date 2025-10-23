from enum import Enum

import pytest
from pydantic import BaseModel, Field, ValidationError

from ask.core.context import ContextASK


class TestContextASK:
    """Test suite for ContextASK base class."""

    def test_to_input_basic_fields(self):
        """Test to_input() with basic field types."""

        class TestModel(ContextASK):
            name: str = Field(description="The name field")
            age: int = Field(description="Age of the person")
            height: float = Field(description="Height in meters")
            is_active: bool = Field(description="Whether active")

        expected = "<name description='The name field'></name>\n<age description='Age of the person'></age>\n<height description='Height in meters'></height>\n<is_active description='Whether active'></is_active>"
        result = TestModel.to_input()
        assert result == expected

    def test_to_input_no_descriptions(self):
        """Test to_input() with fields that have no descriptions raises ValueError."""

        class TestModel(ContextASK):
            name: str
            count: int

        with pytest.raises(ValueError, match="Field name must have a description"):
            TestModel.to_input()

    def test_to_input_mixed_descriptions(self):
        """Test to_input() with some fields having descriptions and others not."""

        class TestModel(ContextASK):
            title: str = Field(description="The title")
            count: int = Field(description="Count of items")
            active: bool = Field(description="Is active status")

        expected = "<title description='The title'></title>\n<count description='Count of items'></count>\n<active description='Is active status'></active>"
        result = TestModel.to_input()
        assert result == expected

    def test_to_input_with_nested_model(self):
        """Test to_input() with nested BaseModel fields."""

        class NestedModel(ContextASK):
            inner_field: str = Field(description="Inner description")
            inner_number: int = Field(description="Inner number field")

        class TestModel(ContextASK):
            name: str = Field(description="Name field")
            nested: NestedModel = Field(description="Nested model")

        expected = "<name description='Name field'></name>\n<nested description='Nested model'>\n <inner_field description='Inner description'></inner_field>\n <inner_number description='Inner number field'></inner_number>\n</nested>"
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
            age: int = Field(description="Age")
            height: float = Field(description="Height")
            is_active: bool = Field(description="Is active")

        instance = TestModel(name="John", age=25, height=5.9, is_active=True)
        expected = "<name>John</name>\n<age>25</age>\n<height>5.9</height>\n<is_active>True</is_active>"
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_none_values(self):
        """Test to_output() with None values."""

        class TestModel(ContextASK):
            name: str = Field(description="Name")
            optional_field: str | None = Field(
                default=None, description="Optional field"
            )

        instance = TestModel(name="Test", optional_field=None)
        expected = "<name>Test</name>\n"
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_list(self):
        """Test to_output() with list field."""

        class TestModel(ContextASK):
            tags: list[str] = Field(description="List of tags")

        instance = TestModel(tags=["python", "test", "mcp"])
        expected = "<tags>python</tags>\n<tags>test</tags>\n<tags>mcp</tags>"
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_empty_list(self):
        """Test to_output() with empty list."""

        class TestModel(ContextASK):
            tags: list[str] = Field(description="Tags")

        instance = TestModel(tags=[])
        expected = ""
        result = instance.to_output()
        assert result == expected

    def test_to_output_with_dict(self):
        """Test to_output() with dictionary field."""

        class TestModel(ContextASK):
            metadata: dict[str, str] = Field(description="Metadata")

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
            status: Status = Field(description="Status")

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
            name: str = Field(description="Person name")
            address: Address = Field(description="Address")

        address = Address(street="123 Main St", city="Anytown")
        person = Person(name="John Doe", address=address)
        expected = "<name>John Doe</name>\n<address>street='123 Main St' city='Anytown'</address>"
        result = person.to_output()
        assert result == expected

    def test_to_output_complex_nested(self):
        """Test to_output() with complex nested structures."""

        class Item(BaseModel):
            name: str
            quantity: int

        class Order(ContextASK):
            order_id: str = Field(description="Order ID")
            items: list[Item] = Field(description="List of items")
            metadata: dict[str, str] = Field(description="Metadata dict")

        order = Order(
            order_id="ORD-001",
            items=[
                Item(name="Widget A", quantity=5),
                Item(name="Widget B", quantity=3),
            ],
            metadata={"priority": "high", "source": "web"},
        )

        result = order.to_output()
        # Check that all parts are present
        assert "<order_id>ORD-001</order_id>" in result
        assert "<items>name='Widget A' quantity=5</items>" in result
        assert "<items>name='Widget B' quantity=3</items>" in result
        assert '<metadata>{"priority": "high", "source": "web"}</metadata>' in result

    def test_to_output_special_characters(self):
        """Test to_output() with special characters that need XML escaping."""

        class TestModel(ContextASK):
            content: str = Field(description="Content")

        instance = TestModel(content="Special chars: <>&\"'")
        expected = "<content>Special chars: &lt;&gt;&amp;\"'</content>"
        result = instance.to_output()
        assert result == expected

    def test_to_output_numeric_types(self):
        """Test to_output() with various numeric types."""

        class TestModel(ContextASK):
            integer_field: int = Field(description="Integer field")
            float_field: float = Field(description="Float field")

        instance = TestModel(integer_field=42, float_field=3.14159)
        result = instance.to_output()
        assert "<integer_field>42</integer_field>" in result
        assert "<float_field>3.14159</float_field>" in result

    def test_to_output_boolean_values(self):
        """Test to_output() with boolean values."""

        class TestModel(ContextASK):
            flag_true: bool = Field(description="True flag")
            flag_false: bool = Field(description="False flag")

        instance = TestModel(flag_true=True, flag_false=False)
        result = instance.to_output()
        assert "<flag_true>True</flag_true>" in result
        assert "<flag_false>False</flag_false>" in result

    def test_inheritance_works(self):
        """Test that ContextASK methods work with inheritance."""

        class BaseModel(ContextASK):
            base_field: str = Field(description="Base field")

        class DerivedModel(BaseModel):
            derived_field: int = Field(description="Derived field")

        # Test class method
        result = DerivedModel.to_input()
        assert "<base_field description='Base field'></base_field>" in result
        assert "<derived_field description='Derived field'></derived_field>" in result

        # Test instance method
        instance = DerivedModel(base_field="test", derived_field=123)
        result = instance.to_output()
        assert "<base_field>test</base_field>" in result
        assert "<derived_field>123</derived_field>" in result

    def test_multiple_instances_independence(self):
        """Test that multiple instances produce independent output."""

        class TestModel(ContextASK):
            value: str = Field(description="Value")

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
            required_field: str = Field(
                description="Required field"
            )  # No default value

        with pytest.raises(ValidationError):
            TestModel()  # Missing required_field

    def test_to_output_invalid_type_raises_error(self):
        """Test that invalid field types raise validation error."""

        class TestModel(ContextASK):
            number_field: int = Field(description="Number field")

        with pytest.raises(ValidationError):
            TestModel(number_field="not_a_number")

    def test_to_input_unknown_field_type(self):
        """Test to_input() with unknown field type annotation."""

        class TestModel(ContextASK):
            custom_field: object = Field(description="Custom field")  # Unknown type

        result = TestModel.to_input()
        assert "<custom_field description='Custom field'></custom_field>" in result

    def test_to_output_none_model_instance(self):
        """Test to_output() behavior with None values in complex fields."""

        class TestModel(ContextASK):
            optional_model: BaseModel | None = Field(
                default=None, description="Optional model"
            )

        instance = TestModel(optional_model=None)
        result = instance.to_output()
        assert result == ""

    def test_to_output_empty_strings(self):
        """Test to_output() with empty string values."""

        class TestModel(ContextASK):
            empty_str: str = Field(description="Empty string")
            whitespace_str: str = Field(description="Whitespace string")

        instance = TestModel(empty_str="", whitespace_str="   ")
        result = instance.to_output()
        assert "<empty_str></empty_str>" in result
        assert "<whitespace_str>   </whitespace_str>" in result

    def test_str_method(self):
        """Test that __str__ returns the same as to_output()."""

        class TestModel(ContextASK):
            name: str = Field(description="Name")

        instance = TestModel(name="test")
        assert str(instance) == instance.to_output()

    def test_to_input_with_default_values(self):
        """Test to_input() with fields that have default values."""

        class TestModel(ContextASK):
            name: str = Field(description="Name")
            count: int = Field(default=10, description="Count")

        result = TestModel.to_input()
        assert "<name description='Name'></name>" in result
        assert "default='10'" in result

    def test_to_input_with_default_factory(self):
        """Test to_input() with fields that have default_factory."""

        class TestModel(ContextASK):
            items: list[str] = Field(default_factory=list, description="Items")

        result = TestModel.to_input()
        assert "<items description='Items' default='[]'></items>" in result
        assert "default='[]'" in result

    def test_to_output_with_default_factory(self):
        """Test to_output() with default_factory field."""

        class TestModel(ContextASK):
            items: list[str] = Field(default_factory=list, description="Items")

        instance = TestModel(items=["default_item"])
        result = instance.to_output()
        assert "<items>default_item</items>" in result

    def test_nested_contextask_models(self):
        """Test to_input() and to_output() with nested ContextASK models."""

        class InnerModel(ContextASK):
            inner_name: str = Field(description="Inner name")

        class OuterModel(ContextASK):
            outer_name: str = Field(description="Outer name")
            inner: InnerModel = Field(description="Inner model")

        # Test to_input
        result = OuterModel.to_input()
        assert "<outer_name description='Outer name'></outer_name>" in result
        assert "<inner description='Inner model'>" in result
        assert "<inner_name description='Inner name'></inner_name>" in result

        # Test to_output
        inner = InnerModel(inner_name="nested")
        outer = OuterModel(outer_name="outer", inner=inner)
        result = outer.to_output()
        assert "<outer_name>outer</outer_name>" in result
        assert "<inner>" in result
        assert "<inner_name>nested</inner_name>" in result

    def test_list_of_contextask(self):
        """Test to_output() with list of ContextASK instances."""

        class ItemModel(ContextASK):
            name: str = Field(description="Item name")

        class ContainerModel(ContextASK):
            items: list[ItemModel] = Field(description="List of items")

        item1 = ItemModel(name="item1")
        item2 = ItemModel(name="item2")
        container = ContainerModel(items=[item1, item2])
        result = container.to_output()
        assert "<items>" in result
        assert "<name>item1</name>" in result
        assert "<name>item2</name>" in result

    def test_union_types(self):
        """Test to_output() with union type fields."""

        class TestModel(ContextASK):
            value: str | int = Field(description="Union value")

        instance1 = TestModel(value="string")
        instance2 = TestModel(value=42)
        result1 = instance1.to_output()
        result2 = instance2.to_output()
        assert "<value>string</value>" in result1
        assert "<value>42</value>" in result2

    def test_deeply_nested_models(self):
        """Test to_input() with deeply nested ContextASK models."""

        class Level3(ContextASK):
            value: str = Field(description="Level 3 value")

        class Level2(ContextASK):
            level3: Level3 = Field(description="Level 3")

        class Level1(ContextASK):
            level2: Level2 = Field(description="Level 2")

        result = Level1.to_input()
        # Check indentation (1 space per level)
        assert "<level2 description='Level 2'>" in result
        assert " <level3 description='Level 3'>" in result
        assert "  <value description='Level 3 value'></value>" in result

    def test_xml_escaping_in_descriptions(self):
        """Test XML escaping in field descriptions."""

        class TestModel(ContextASK):
            field: str = Field(description="Description with <>&")

        result = TestModel.to_input()
        assert "description='Description with &lt;&gt;&amp;'" in result

    def test_model_with_only_optional_fields(self):
        """Test model with only optional fields."""

        class TestModel(ContextASK):
            optional1: str | None = Field(default=None, description="Optional 1")
            optional2: int | None = Field(default=None, description="Optional 2")

        instance = TestModel()
        result = instance.to_output()
        assert result == "\n"

        instance2 = TestModel(optional1="test")
        result2 = instance2.to_output()
        assert "<optional1>test</optional1>" in result2
        assert "<optional2>" not in result2
